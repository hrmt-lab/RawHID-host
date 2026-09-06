use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    ffi::CString,
    fmt,
    time::{Duration, Instant},
};

use hidapi::{BusType, HidApi};
use thiserror::Error;
use tracing::{debug, warn};

use crate::{
    config::HidConfig,
    packet::{
        AiClientStatePacket, AiUsagePacket, ComboInfo, ComboItem, ConfigRequest, ConfigResponse,
        ConfigStatus, DeviceHello, EncoderBinding, EncoderGetBindings, EncoderGetInfo, Packet,
        TimeSyncPacket, UplinkPacket, CAPABILITY_AI_CLIENT_CLAUDE_CODE,
        CAPABILITY_AI_CLIENT_DISPLAY_SLOT, CAPABILITY_AI_CLIENT_SCREENKEY_STATE,
        CAPABILITY_AI_CLIENT_STATE, CAPABILITY_AI_CLIENT_WORK_PHASE, CAPABILITY_CONFIG_RPC,
        PACKET_SIZE, REPORT_SIZE,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceConnectionType {
    Usb,
    Bluetooth,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeviceInfo {
    pub path: String,
    pub vendor_id: u16,
    pub product_id: u16,
    pub usage_page: u16,
    pub usage: u16,
    pub connection_type: DeviceConnectionType,
    pub manufacturer: Option<String>,
    pub product: Option<String>,
    pub serial_number: Option<String>,
    pub capabilities: u32,
    #[serde(
        serialize_with = "serialize_device_uid_hash",
        deserialize_with = "deserialize_device_uid_hash"
    )]
    pub device_uid_hash: Option<u64>,
}
fn serialize_device_uid_hash<S>(value: &Option<u64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        Some(uid) => serializer.serialize_some(&format!("uid:{uid:016x}")),
        None => serializer.serialize_none(),
    }
}

fn deserialize_device_uid_hash<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <Option<String> as serde::Deserialize>::deserialize(deserializer)?;
    value
        .map(|value| {
            let hex = value.strip_prefix("uid:").unwrap_or(&value);
            u64::from_str_radix(hex, 16).map_err(serde::de::Error::custom)
        })
        .transpose()
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ProbeResult {
    pub device: DeviceInfo,
    pub verified: bool,
    pub error: Option<String>,
}

pub trait HidTransport {
    fn candidates(&self, usage_page: u16, usage: u16) -> Result<Vec<DeviceInfo>, HidError>;
    fn hello(
        &self,
        device: &DeviceInfo,
        packet: Packet,
        timeout_ms: i32,
    ) -> Result<Option<DeviceHello>, HidError>;
    fn write_report(&self, device: &DeviceInfo, report: &[u8; REPORT_SIZE])
        -> Result<(), HidError>;
    /// Read one 64-byte input report if available. The default no-op keeps
    /// transports without an uplink path (and test mocks) working unchanged.
    fn read_packet(
        &self,
        _device: &DeviceInfo,
        _timeout_ms: i32,
    ) -> Result<Option<[u8; PACKET_SIZE]>, HidError> {
        Ok(None)
    }
    fn forget_device(&self, _device: &DeviceInfo) {}
    /// Close only the OS handle. Device verification and packets already read
    /// from the transport must remain intact.
    fn close_device_handle(&self, _device: &DeviceInfo) {}
}

pub struct RealHidTransport {
    api: RefCell<HidApi>,
    handles: RefCell<HashMap<String, hidapi::HidDevice>>,
    /// Uplink packets that arrived while waiting for a DEVICE_HELLO response;
    /// they are handed back out through `read_packet` instead of being lost.
    pending_uplink: RefCell<HashMap<String, VecDeque<[u8; PACKET_SIZE]>>>,
}

impl RealHidTransport {
    pub fn new() -> Result<Self, HidError> {
        Ok(Self {
            api: RefCell::new(HidApi::new().map_err(HidError::Hid)?),
            handles: RefCell::new(HashMap::new()),
            pending_uplink: RefCell::new(HashMap::new()),
        })
    }

    fn open_device(&self, path: &str) -> Result<hidapi::HidDevice, HidError> {
        let path = CString::new(path).map_err(|_| HidError::InvalidDevicePath)?;
        self.api.borrow().open_path(&path).map_err(HidError::Hid)
    }
}

const BLE_HID_SERVICE_UUID: &str = "00001812-0000-1000-8000-00805f9b34fb";

fn connection_type_from_hid(device: &hidapi::DeviceInfo) -> DeviceConnectionType {
    match device.bus_type() {
        BusType::Usb => DeviceConnectionType::Usb,
        BusType::Bluetooth => DeviceConnectionType::Bluetooth,
        _ => {
            let path = device.path().to_string_lossy().to_ascii_lowercase();
            if path.contains(BLE_HID_SERVICE_UUID) {
                DeviceConnectionType::Bluetooth
            } else if path.contains("hid#vid_") || path.contains("hid#vid&") {
                DeviceConnectionType::Usb
            } else {
                DeviceConnectionType::Unknown
            }
        }
    }
}

impl fmt::Debug for RealHidTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealHidTransport")
            .field("open_handles", &self.handles.borrow().len())
            .finish_non_exhaustive()
    }
}

impl HidTransport for RealHidTransport {
    fn candidates(&self, usage_page: u16, usage: u16) -> Result<Vec<DeviceInfo>, HidError> {
        let mut api = self.api.borrow_mut();
        api.refresh_devices().map_err(HidError::Hid)?;
        Ok(api
            .device_list()
            .filter(|device| device.usage_page() == usage_page && device.usage() == usage)
            .map(|device| DeviceInfo {
                path: device.path().to_string_lossy().to_string(),
                vendor_id: device.vendor_id(),
                product_id: device.product_id(),
                usage_page: device.usage_page(),
                usage: device.usage(),
                connection_type: connection_type_from_hid(device),
                manufacturer: device.manufacturer_string().map(ToString::to_string),
                product: device.product_string().map(ToString::to_string),
                serial_number: device.serial_number().map(ToString::to_string),
                capabilities: 0,
                device_uid_hash: None,
            })
            .collect())
    }

    fn hello(
        &self,
        device: &DeviceInfo,
        packet: Packet,
        timeout_ms: i32,
    ) -> Result<Option<DeviceHello>, HidError> {
        {
            let mut handles = self.handles.borrow_mut();
            if !handles.contains_key(&device.path) {
                let hid = self.open_device(&device.path)?;
                handles.insert(device.path.clone(), hid);
            }
        }

        let handles = self.handles.borrow();
        let hid = handles
            .get(&device.path)
            .expect("handle inserted above is present");
        hid.write(&packet.encode_report()).map_err(HidError::Hid)?;

        // Devices may emit uplink packets at any time, so the HELLO response
        // is not necessarily the next report. Skip (and keep) other valid HL
        // packets until the matching DEVICE_HELLO arrives or the timeout ends.
        let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(0) as u64);
        let mut buffer = [0u8; PACKET_SIZE];
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let read = hid
                .read_timeout(&mut buffer, (remaining.as_millis() as i32).max(1))
                .map_err(HidError::Hid)?;
            if read == PACKET_SIZE {
                match DeviceHello::decode_payload(&buffer) {
                    Ok(response) if response.seq == packet.seq => {
                        return Ok(Some(response));
                    }
                    Ok(_) => {} // stale HELLO from an earlier probe; keep waiting
                    Err(hello_error) => match UplinkPacket::decode_payload(&buffer) {
                        Ok(_) => {
                            self.pending_uplink
                                .borrow_mut()
                                .entry(device.path.clone())
                                .or_default()
                                .push_back(buffer);
                        }
                        Err(uplink_error) => {
                            debug!(
                                "dropping non-HELLO packet while waiting for HELLO from {}: hello={}, uplink={}",
                                device.path, hello_error, uplink_error
                            );
                        }
                    },
                }
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
        }
    }

    fn write_report(
        &self,
        device: &DeviceInfo,
        report: &[u8; REPORT_SIZE],
    ) -> Result<(), HidError> {
        if let Some(hid) = self.handles.borrow().get(&device.path) {
            match hid.write(report) {
                Ok(_) => return Ok(()),
                Err(error) => {
                    debug!(
                        "Raw HID write failed for {}; reopening handle and retrying once: {}",
                        device.path, error
                    );
                }
            }
        }

        self.handles.borrow_mut().remove(&device.path);
        let hid = self.open_device(&device.path)?;
        hid.write(report).map_err(HidError::Hid)?;
        self.handles.borrow_mut().insert(device.path.clone(), hid);
        Ok(())
    }

    fn read_packet(
        &self,
        device: &DeviceInfo,
        timeout_ms: i32,
    ) -> Result<Option<[u8; PACKET_SIZE]>, HidError> {
        if let Some(queue) = self.pending_uplink.borrow_mut().get_mut(&device.path) {
            if let Some(buffered) = queue.pop_front() {
                return Ok(Some(buffered));
            }
        }

        let mut handles = self.handles.borrow_mut();
        if !handles.contains_key(&device.path) {
            let hid = self.open_device(&device.path)?;
            handles.insert(device.path.clone(), hid);
        }
        let hid = handles
            .get(&device.path)
            .expect("handle inserted above is present");

        let mut buffer = [0u8; PACKET_SIZE];
        let read = hid
            .read_timeout(&mut buffer, timeout_ms)
            .map_err(HidError::Hid)?;
        if read == PACKET_SIZE {
            Ok(Some(buffer))
        } else {
            Ok(None)
        }
    }

    fn forget_device(&self, device: &DeviceInfo) {
        self.handles.borrow_mut().remove(&device.path);
        self.pending_uplink.borrow_mut().remove(&device.path);
    }

    fn close_device_handle(&self, device: &DeviceInfo) {
        self.handles.borrow_mut().remove(&device.path);
    }
}

#[derive(Debug)]
pub struct HidDeviceManager<T = RealHidTransport> {
    transport: T,
    config: HidConfig,
    verified: Vec<DeviceInfo>,
    missed_probe_counts: HashMap<String, u8>,
    probed_candidate_paths: HashSet<String>,
    unresponsive_logged_paths: HashSet<String>,
    seq: u8,
    generation: u64,
    last_probe_at: Option<Instant>,
    deferred_uplink: VecDeque<(DeviceInfo, UplinkPacket)>,
}

impl HidDeviceManager<RealHidTransport> {
    pub fn real(config: HidConfig) -> Result<Self, HidError> {
        Ok(Self::new(config, RealHidTransport::new()?))
    }
}

impl<T: HidTransport> HidDeviceManager<T> {
    pub fn new(config: HidConfig, transport: T) -> Self {
        Self {
            transport,
            config,
            verified: Vec::new(),
            missed_probe_counts: HashMap::new(),
            probed_candidate_paths: HashSet::new(),
            unresponsive_logged_paths: HashSet::new(),
            seq: 0,
            generation: 0,
            last_probe_at: None,
            deferred_uplink: VecDeque::new(),
        }
    }

    pub fn verified_devices(&self) -> &[DeviceInfo] {
        &self.verified
    }

    pub fn device_generation(&self) -> u64 {
        self.generation
    }

    pub fn update_config(&mut self, config: HidConfig) {
        let identity_changed =
            self.config.usage_page != config.usage_page || self.config.usage != config.usage;
        self.config = config;
        self.last_probe_at = None;
        if identity_changed {
            for device in &self.verified {
                self.transport.forget_device(device);
            }
            self.verified.clear();
            self.missed_probe_counts.clear();
            self.probed_candidate_paths.clear();
            self.unresponsive_logged_paths.clear();
            self.deferred_uplink.clear();
            self.generation = self.generation.wrapping_add(1);
        }
    }

    pub fn probe(&mut self) -> Result<Vec<ProbeResult>, HidError> {
        self.probe_at(Instant::now())
    }

    fn probe_at(&mut self, now: Instant) -> Result<Vec<ProbeResult>, HidError> {
        let candidates = self
            .transport
            .candidates(self.config.usage_page, self.config.usage)?;
        let candidate_paths: HashSet<String> = candidates
            .iter()
            .map(|device| device.path.clone())
            .collect();
        let mut results = Vec::with_capacity(candidates.len());
        let mut verified = Vec::new();

        const INITIAL_HELLO_ATTEMPTS: usize = 3;
        for device in candidates {
            let attempts = if self.probed_candidate_paths.insert(device.path.clone()) {
                INITIAL_HELLO_ATTEMPTS
            } else {
                1
            };
            let mut hello_result = Ok(None);
            for _ in 0..attempts {
                let seq = self.next_seq();
                let hello = Packet::host_hello(seq);
                hello_result = self
                    .transport
                    .hello(&device, hello, self.config.hello_timeout_ms);
                if matches!(hello_result, Ok(Some(_))) {
                    break;
                }
            }
            match hello_result {
                Ok(Some(hello)) => {
                    debug!("Raw HID HELLO succeeded for {}", device.path);
                    let mut verified_device = device.clone();
                    verified_device.capabilities = hello.capabilities;
                    verified_device.device_uid_hash = hello.device_uid_hash;
                    self.missed_probe_counts.remove(&verified_device.path);
                    self.unresponsive_logged_paths.remove(&verified_device.path);
                    upsert_verified_by_identity(&mut verified, verified_device.clone());
                    results.push(ProbeResult {
                        device: verified_device,
                        verified: true,
                        error: None,
                    });
                }
                Ok(None) => {
                    if self.unresponsive_logged_paths.insert(device.path.clone()) {
                        warn!(
                            "Raw HID HELLO did not receive DEVICE_HELLO after {attempts} attempt(s) for {}",
                            device.path
                        );
                    }
                    results.push(ProbeResult {
                        device,
                        verified: false,
                        error: None,
                    });
                }
                Err(error) => {
                    let message = error.to_string();
                    results.push(ProbeResult {
                        device,
                        verified: false,
                        error: Some(message),
                    });
                }
            }
        }

        const MAX_TRANSIENT_HELLO_MISSES: u8 = 2;
        for old_device in &self.verified {
            if verified.iter().any(|device| device.path == old_device.path) {
                continue;
            }
            if !candidate_paths.contains(&old_device.path) {
                continue;
            }

            let misses = self
                .missed_probe_counts
                .entry(old_device.path.clone())
                .and_modify(|count| *count = count.saturating_add(1))
                .or_insert(1);
            if *misses <= MAX_TRANSIENT_HELLO_MISSES {
                debug!(
                    "keeping previously verified Raw HID device {} after transient HELLO miss {}",
                    old_device.path, misses
                );
                verified.push(old_device.clone());
            }
        }

        self.probed_candidate_paths
            .retain(|path| candidate_paths.contains(path));
        self.unresponsive_logged_paths
            .retain(|path| candidate_paths.contains(path));

        for old_device in &self.verified {
            if !verified.iter().any(|device| device.path == old_device.path) {
                self.transport.forget_device(old_device);
                self.missed_probe_counts.remove(&old_device.path);
            }
        }

        if self.verified != verified {
            self.generation = self.generation.wrapping_add(1);
        }
        self.verified = verified;
        self.last_probe_at = Some(now);
        Ok(results)
    }

    pub fn ensure_verified(&mut self) -> Result<(), HidError> {
        self.ensure_verified_at(Instant::now())
    }

    fn ensure_verified_at(&mut self, now: Instant) -> Result<(), HidError> {
        let rescan_interval = Duration::from_secs(self.config.rescan_interval_sec.max(1));
        let rescan_due = match self.last_probe_at {
            Some(last_probe_at) => now.duration_since(last_probe_at) >= rescan_interval,
            None => true,
        };
        if rescan_due {
            let _ = self.probe_at(now)?;
        }
        Ok(())
    }

    pub fn send_set_layer(&mut self, layer: u8) -> Result<usize, HidError> {
        let packet = Packet::set_layer(layer, self.next_seq()).map_err(HidError::Packet)?;
        self.send_report_to_verified(packet.encode_report())
    }

    pub fn send_clear(&mut self) -> Result<usize, HidError> {
        let packet = Packet::clear(self.next_seq());
        self.send_report_to_verified(packet.encode_report())
    }

    pub fn send_ai_client_state(
        &mut self,
        state: AiClientStatePacket,
        work_phase_only: bool,
    ) -> Result<usize, HidError> {
        self.ensure_verified()?;
        let seq = self.next_seq();
        let legacy_report = state.encode_report(seq);
        let work_phase_report = state.encode_work_phase_report(seq);
        let display_slot_report = state.encode_display_slot_report(seq);
        let screenkey_state_report = state.encode_screenkey_state_report(seq);
        let mut sent = 0usize;
        let previous_len = self.verified.len();
        let mut retained = Vec::with_capacity(self.verified.len());

        for device in self.verified.drain(..) {
            let supports_display_slot = device.capabilities
                & (CAPABILITY_AI_CLIENT_DISPLAY_SLOT | CAPABILITY_AI_CLIENT_WORK_PHASE)
                == (CAPABILITY_AI_CLIENT_DISPLAY_SLOT | CAPABILITY_AI_CLIENT_WORK_PHASE);
            // The 9-byte payload is a strict superset of the 8-byte one, so
            // it is only sent to a device that advertises all three of bit
            // 14, bit 13, and bit 11 -- never bit 14 alone.
            let supports_screenkey_state = supports_display_slot
                && device.capabilities & CAPABILITY_AI_CLIENT_SCREENKEY_STATE != 0;
            if device.capabilities & CAPABILITY_AI_CLIENT_STATE == 0
                || (state.display_slot != 0 && !supports_display_slot)
                || (state.client_type == crate::packet::AiClientType::ClaudeCode
                    && device.capabilities & CAPABILITY_AI_CLIENT_CLAUDE_CODE == 0)
                || (work_phase_only && device.capabilities & CAPABILITY_AI_CLIENT_WORK_PHASE == 0)
            {
                retained.push(device);
                continue;
            }
            let report = if supports_screenkey_state {
                &screenkey_state_report
            } else if supports_display_slot {
                &display_slot_report
            } else if device.capabilities & CAPABILITY_AI_CLIENT_WORK_PHASE != 0 {
                &work_phase_report
            } else {
                &legacy_report
            };
            match self.transport.write_report(&device, report) {
                Ok(()) => {
                    sent += 1;
                    retained.push(device);
                }
                Err(error) => {
                    warn!("Raw HID write failed for {}: {}", device.path, error);
                    self.transport.forget_device(&device);
                }
            }
        }

        if previous_len != retained.len() {
            self.generation = self.generation.wrapping_add(1);
        }
        self.verified = retained;
        Ok(sent)
    }

    pub fn send_set_layer_to_device(
        &mut self,
        device: &DeviceInfo,
        layer: u8,
    ) -> Result<(), HidError> {
        let packet = Packet::set_layer(layer, self.next_seq()).map_err(HidError::Packet)?;
        self.send_report_to_device(device, packet.encode_report())
    }

    pub fn send_clear_to_device(&mut self, device: &DeviceInfo) -> Result<(), HidError> {
        let packet = Packet::clear(self.next_seq());
        self.send_report_to_device(device, packet.encode_report())
    }

    pub fn send_time_sync(&mut self, packet: TimeSyncPacket) -> Result<usize, HidError> {
        self.send_report_to_verified(packet.encode_report())
    }

    pub fn send_ai_usage(&mut self, packet: AiUsagePacket) -> Result<usize, HidError> {
        self.send_report_to_verified(packet.encode_report())
    }

    pub fn config_get_encoder_info(
        &mut self,
        device: &DeviceInfo,
    ) -> Result<EncoderGetInfo, HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }

        let request = ConfigRequest::encoder_get_info(self.next_seq());
        let response = self.send_config_request_with_retry(device, request, "ENCODER GET_INFO")?;
        if response.status == ConfigStatus::Ok {
            return response
                .encoder_get_info
                .ok_or(HidError::ConfigRpcMissingPayload);
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    pub fn config_get_encoder_bindings(
        &mut self,
        device: &DeviceInfo,
        layer_id: u32,
        encoder_id: u8,
    ) -> Result<EncoderGetBindings, HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }

        let request = ConfigRequest::encoder_get_bindings(self.next_seq(), layer_id, encoder_id);
        let response =
            self.send_config_request_with_retry(device, request, "ENCODER GET_BINDINGS")?;
        if response.status == ConfigStatus::Ok {
            return response
                .encoder_get_bindings
                .ok_or(HidError::ConfigRpcMissingPayload);
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    pub fn config_set_encoder_bindings(
        &mut self,
        device: &DeviceInfo,
        layer_id: u32,
        encoder_id: u8,
        cw_binding: EncoderBinding,
        ccw_binding: EncoderBinding,
    ) -> Result<(), HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }

        let request = ConfigRequest::encoder_set_bindings(
            self.next_seq(),
            layer_id,
            encoder_id,
            cw_binding,
            ccw_binding,
        );
        let response =
            self.send_config_request_with_retry(device, request, "ENCODER SET_BINDINGS")?;
        if response.status == ConfigStatus::Ok {
            return Ok(());
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    pub fn config_get_encoder_dirty(&mut self, device: &DeviceInfo) -> Result<bool, HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }

        let request = ConfigRequest::encoder_get_dirty(self.next_seq());
        let response = self.send_config_request_with_retry(device, request, "ENCODER GET_DIRTY")?;
        if response.status == ConfigStatus::Ok {
            return response
                .encoder_get_dirty
                .ok_or(HidError::ConfigRpcMissingPayload);
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    pub fn config_save_encoder(&mut self, device: &DeviceInfo) -> Result<(), HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }

        let request = ConfigRequest::encoder_save(self.next_seq());
        self.send_config_status_request(device, request, "ENCODER SAVE")
    }

    pub fn config_discard_encoder(&mut self, device: &DeviceInfo) -> Result<(), HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }

        let request = ConfigRequest::encoder_discard(self.next_seq());
        self.send_config_status_request(device, request, "ENCODER DISCARD")
    }

    pub fn config_clear_encoder_override(
        &mut self,
        device: &DeviceInfo,
        layer_id: u32,
        encoder_id: u8,
    ) -> Result<(), HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }

        let request = ConfigRequest::encoder_clear_override(self.next_seq(), layer_id, encoder_id);
        self.send_config_status_request(device, request, "ENCODER CLEAR_OVERRIDE")
    }

    pub fn config_get_combo_info(&mut self, device: &DeviceInfo) -> Result<ComboInfo, HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_get_info(self.next_seq());
        let response = self.send_config_request_with_retry(device, request, "COMBO GET_INFO")?;
        if response.status == ConfigStatus::Ok {
            return response.combo_info.ok_or(HidError::ConfigRpcMissingPayload);
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    pub fn config_get_combo(
        &mut self,
        device: &DeviceInfo,
        slot: u8,
    ) -> Result<ComboItem, HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_get_combo(self.next_seq(), slot)?;
        let response = self.send_config_request_with_retry(device, request, "COMBO GET_COMBO")?;
        if response.status == ConfigStatus::Ok {
            return response.combo_item.ok_or(HidError::ConfigRpcMissingPayload);
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    pub fn config_set_combo(
        &mut self,
        device: &DeviceInfo,
        item: ComboItem,
    ) -> Result<(), HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_set_combo(self.next_seq(), item)?;
        self.send_config_status_request(device, request, "COMBO SET_COMBO")
    }

    pub fn config_get_combo_dirty(&mut self, device: &DeviceInfo) -> Result<bool, HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_get_dirty(self.next_seq());
        let response = self.send_config_request_with_retry(device, request, "COMBO GET_DIRTY")?;
        if response.status == ConfigStatus::Ok {
            return response
                .combo_dirty
                .ok_or(HidError::ConfigRpcMissingPayload);
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    pub fn config_save_combos(&mut self, device: &DeviceInfo) -> Result<(), HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_save(self.next_seq());
        // Combo SAVE completes on the firmware workqueue after the USB callback
        // returns. Keep the ordinary Config RPC timeout for every other op, but
        // allow enough room for one NVS table write before retrying the same request.
        const COMBO_SAVE_TIMEOUT_MS: i32 = 1_500;
        let timeout_ms = self.config.hello_timeout_ms.max(COMBO_SAVE_TIMEOUT_MS);
        let response =
            self.send_config_request_with_retry_timeout(device, request, "COMBO SAVE", timeout_ms)?;
        if response.status == ConfigStatus::Ok {
            Ok(())
        } else {
            Err(HidError::ConfigRpcStatus(response.status))
        }
    }

    pub fn config_discard_combos(&mut self, device: &DeviceInfo) -> Result<(), HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_discard(self.next_seq());
        self.send_config_status_request(device, request, "COMBO DISCARD")
    }

    pub fn config_delete_combo(&mut self, device: &DeviceInfo, slot: u8) -> Result<(), HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_delete_combo(self.next_seq(), slot)?;
        self.send_config_status_request(device, request, "COMBO DELETE_COMBO")
    }

    pub fn config_reset_combos_to_keymap(&mut self, device: &DeviceInfo) -> Result<(), HidError> {
        self.require_config_rpc(device)?;
        let request = ConfigRequest::combo_reset_to_keymap(self.next_seq());
        self.send_config_status_request(device, request, "COMBO RESET_TO_KEYMAP")
    }

    fn require_config_rpc(&self, device: &DeviceInfo) -> Result<(), HidError> {
        if device.capabilities & CAPABILITY_CONFIG_RPC == 0 {
            return Err(HidError::ConfigRpcUnsupported);
        }
        Ok(())
    }

    fn send_config_status_request(
        &mut self,
        device: &DeviceInfo,
        request: ConfigRequest,
        label: &str,
    ) -> Result<(), HidError> {
        let response = self.send_config_request_with_retry(device, request, label)?;
        if response.status == ConfigStatus::Ok {
            return Ok(());
        }
        Err(HidError::ConfigRpcStatus(response.status))
    }

    fn send_config_request_with_retry(
        &mut self,
        device: &DeviceInfo,
        request: ConfigRequest,
        label: &str,
    ) -> Result<ConfigResponse, HidError> {
        self.send_config_request_with_retry_timeout(
            device,
            request,
            label,
            self.config.hello_timeout_ms,
        )
    }

    fn send_config_request_with_retry_timeout(
        &mut self,
        device: &DeviceInfo,
        request: ConfigRequest,
        label: &str,
        timeout_ms: i32,
    ) -> Result<ConfigResponse, HidError> {
        // Windows hidapi can leave the handle in an overlapped I/O state after
        // the HELLO/probe read path. Config RPC is synchronous request/response,
        // so reopen once before the first write to keep it isolated from probe.
        self.transport.close_device_handle(device);

        const MAX_ATTEMPTS: usize = 2;
        for attempt in 0..MAX_ATTEMPTS {
            self.send_report_to_device(device, request.encode_report())?;
            match self.wait_for_config_response(device, request, timeout_ms)? {
                Some(response) => return Ok(response),
                None if attempt + 1 < MAX_ATTEMPTS => {
                    debug!(
                        "Config RPC {label} timed out for {}; retrying with seq {}",
                        device.path, request.seq
                    );
                }
                None => return Err(HidError::ConfigRpcTimeout),
            }
        }
        Err(HidError::ConfigRpcTimeout)
    }

    fn wait_for_config_response(
        &mut self,
        device: &DeviceInfo,
        request: ConfigRequest,
        timeout_ms: i32,
    ) -> Result<Option<ConfigResponse>, HidError> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(1) as u64);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let timeout_ms = (remaining.as_millis() as i32).max(1);
            let Some(payload) = self.transport.read_packet(device, timeout_ms)? else {
                if Instant::now() >= deadline {
                    return Ok(None);
                }
                continue;
            };

            match ConfigResponse::decode_payload(&payload) {
                Ok(response) if response.is_response_to(request) => return Ok(Some(response)),
                Ok(response) => {
                    debug!(
                        "ignoring stale Config RPC response from {}: seq={} feature={:#04x} op={:#04x}",
                        device.path, response.seq, response.feature, response.op
                    );
                }
                Err(config_error) => match UplinkPacket::decode_payload(&payload) {
                    Ok(packet) => {
                        debug!(
                            "uplink while waiting for Config RPC from {}: {:?}",
                            device.path,
                            packet.packet_type()
                        );
                        self.deferred_uplink.push_back((device.clone(), packet));
                    }
                    Err(uplink_error) => {
                        debug!(
                            "dropping packet while waiting for Config RPC from {}: config={}, uplink={}",
                            device.path, config_error, uplink_error
                        );
                    }
                },
            }
        }
    }

    fn send_report_to_device(
        &mut self,
        device: &DeviceInfo,
        report: [u8; REPORT_SIZE],
    ) -> Result<(), HidError> {
        self.transport
            .write_report(device, &report)
            .inspect_err(|_| {
                self.transport.forget_device(device);
                self.verified.retain(|d| d.path != device.path);
                self.generation = self.generation.wrapping_add(1);
            })
    }

    pub fn send_report_to_verified(
        &mut self,
        report: [u8; REPORT_SIZE],
    ) -> Result<usize, HidError> {
        self.ensure_verified()?;
        let mut sent = 0usize;
        let previous_len = self.verified.len();
        let mut retained = Vec::with_capacity(self.verified.len());

        for device in self.verified.drain(..) {
            match self.transport.write_report(&device, &report) {
                Ok(()) => {
                    sent += 1;
                    retained.push(device);
                }
                Err(error) => {
                    warn!("Raw HID write failed for {}: {}", device.path, error);
                    self.transport.forget_device(&device);
                }
            }
        }

        if previous_len != retained.len() {
            self.generation = self.generation.wrapping_add(1);
        }
        self.verified = retained;
        Ok(sent)
    }

    /// Drain pending device-initiated packets from all verified devices.
    /// Invalid packets are logged and skipped; read errors drop the device
    /// (same policy as write failures).
    pub fn drain_uplink(&mut self) -> Vec<(DeviceInfo, UplinkPacket)> {
        // Livelock guard: a chattering device cannot pin the monitor loop.
        const MAX_PACKETS_PER_DEVICE: usize = 64;

        let devices = self.verified.clone();
        let mut events: Vec<_> = self.deferred_uplink.drain(..).collect();
        for device in devices {
            for _ in 0..MAX_PACKETS_PER_DEVICE {
                match self.transport.read_packet(&device, 0) {
                    Ok(Some(payload)) => match UplinkPacket::decode_payload(&payload) {
                        Ok(packet) => events.push((device.clone(), packet)),
                        Err(error) => {
                            warn!("invalid uplink packet from {}: {}", device.path, error);
                        }
                    },
                    Ok(None) => break,
                    Err(error) => {
                        warn!("Raw HID read failed for {}: {}", device.path, error);
                        self.transport.forget_device(&device);
                        self.verified.retain(|d| d.path != device.path);
                        self.generation = self.generation.wrapping_add(1);
                        break;
                    }
                }
            }
        }
        events
    }

    fn next_seq(&mut self) -> u8 {
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        seq
    }
}

fn upsert_verified_by_identity(verified: &mut Vec<DeviceInfo>, device: DeviceInfo) {
    let existing = match device.device_uid_hash {
        Some(uid) => verified
            .iter()
            .position(|candidate| candidate.device_uid_hash == Some(uid)),
        None => verified
            .iter()
            .position(|candidate| candidate.path == device.path),
    };
    if let Some(index) = existing {
        verified[index].capabilities = device.capabilities;
    } else {
        verified.push(device);
    }
}

#[derive(Debug, Error)]
pub enum HidError {
    #[error("HID error: {0}")]
    Hid(#[from] hidapi::HidError),
    #[error("packet error: {0}")]
    Packet(#[from] crate::packet::PacketError),
    #[error("invalid HID device path")]
    InvalidDevicePath,
    #[error("device does not advertise CONFIG_RPC capability")]
    ConfigRpcUnsupported,
    #[error("Host Link device was not found")]
    DeviceNotFound,
    #[error("Config RPC request timed out")]
    ConfigRpcTimeout,
    #[error("Config RPC failed with status {0:?}")]
    ConfigRpcStatus(ConfigStatus),
    #[error("Config RPC OK response did not contain the expected payload")]
    ConfigRpcMissingPayload,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, collections::HashSet};

    #[derive(Debug, Default)]
    struct MockTransport {
        candidates: RefCell<Vec<DeviceInfo>>,
        hello_paths: RefCell<HashSet<String>>,
        hello_capabilities: RefCell<HashMap<String, u32>>,
        hello_uids: RefCell<HashMap<String, Option<u64>>>,
        hello_failures_remaining: RefCell<HashMap<String, usize>>,
        hello_calls: RefCell<Vec<(String, u8, i32)>>,
        failing_writes: RefCell<HashSet<String>>,
        failing_reads: RefCell<HashSet<String>>,
        writes: RefCell<Vec<(String, [u8; REPORT_SIZE])>>,
        uplink: RefCell<HashMap<String, VecDeque<[u8; PACKET_SIZE]>>>,
    }

    impl HidTransport for MockTransport {
        fn candidates(&self, _usage_page: u16, _usage: u16) -> Result<Vec<DeviceInfo>, HidError> {
            Ok(self.candidates.borrow().clone())
        }

        fn hello(
            &self,
            device: &DeviceInfo,
            packet: Packet,
            _timeout_ms: i32,
        ) -> Result<Option<DeviceHello>, HidError> {
            self.hello_calls
                .borrow_mut()
                .push((device.path.clone(), packet.seq, _timeout_ms));
            if let Some(remaining) = self
                .hello_failures_remaining
                .borrow_mut()
                .get_mut(&device.path)
            {
                if *remaining > 0 {
                    *remaining -= 1;
                    return Ok(None);
                }
            }
            Ok(self
                .hello_paths
                .borrow()
                .contains(&device.path)
                .then_some(DeviceHello {
                    seq: packet.seq,
                    capabilities: self
                        .hello_capabilities
                        .borrow()
                        .get(&device.path)
                        .copied()
                        .unwrap_or(crate::packet::CAPABILITY_APP_LAYER),
                    device_uid_hash: self
                        .hello_uids
                        .borrow()
                        .get(&device.path)
                        .copied()
                        .unwrap_or(device.device_uid_hash),
                }))
        }

        fn write_report(
            &self,
            device: &DeviceInfo,
            report: &[u8; REPORT_SIZE],
        ) -> Result<(), HidError> {
            if self.failing_writes.borrow().contains(&device.path) {
                return Err(HidError::InvalidDevicePath);
            }
            self.writes
                .borrow_mut()
                .push((device.path.clone(), *report));
            Ok(())
        }

        fn read_packet(
            &self,
            device: &DeviceInfo,
            _timeout_ms: i32,
        ) -> Result<Option<[u8; PACKET_SIZE]>, HidError> {
            if self.failing_reads.borrow().contains(&device.path) {
                return Err(HidError::InvalidDevicePath);
            }
            Ok(self
                .uplink
                .borrow_mut()
                .get_mut(&device.path)
                .and_then(VecDeque::pop_front))
        }
    }

    fn device(path: &str) -> DeviceInfo {
        DeviceInfo {
            path: path.to_string(),
            vendor_id: 1,
            product_id: 2,
            usage_page: 0xFF60,
            usage: 0x61,
            connection_type: DeviceConnectionType::Usb,
            manufacturer: None,
            product: None,
            serial_number: None,
            capabilities: crate::packet::CAPABILITY_APP_LAYER,
            device_uid_hash: path.as_bytes().first().copied().map(u64::from),
        }
    }

    fn device_with_capabilities(path: &str, capabilities: u32) -> DeviceInfo {
        let mut device = device(path);
        device.capabilities = capabilities;
        device
    }

    #[test]
    fn probe_keeps_only_hello_devices() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a"), device("b")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("b".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);

        let results = manager.probe().unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(manager.verified_devices(), &[device("b")]);
    }

    #[test]
    fn initial_probe_retries_hello_three_times_with_fresh_sequences() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        transport
            .hello_failures_remaining
            .borrow_mut()
            .insert("a".to_string(), 2);
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);

        let results = manager.probe().unwrap();

        assert!(results[0].verified);
        assert_eq!(manager.verified_devices(), &[device("a")]);
        assert_eq!(
            manager.transport.hello_calls.borrow().as_slice(),
            &[
                ("a".to_string(), 0, 500),
                ("a".to_string(), 1, 500),
                ("a".to_string(), 2, 500)
            ]
        );
    }

    #[test]
    fn unresponsive_candidate_gets_one_hello_on_periodic_retry() {
        let start = Instant::now();
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);

        manager.probe_at(start).unwrap();
        assert_eq!(manager.transport.hello_calls.borrow().len(), 3);
        manager.transport.hello_calls.borrow_mut().clear();
        manager
            .transport
            .hello_paths
            .borrow_mut()
            .insert("a".to_string());

        manager
            .ensure_verified_at(start + Duration::from_secs(4))
            .unwrap();
        assert!(manager.transport.hello_calls.borrow().is_empty());

        manager
            .ensure_verified_at(start + Duration::from_secs(5))
            .unwrap();
        assert_eq!(manager.transport.hello_calls.borrow().len(), 1);
        assert_eq!(manager.verified_devices(), &[device("a")]);
    }

    #[test]
    fn duplicate_device_uid_is_registered_once_and_updates_capabilities() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a"), device("b")]),
            ..MockTransport::default()
        };
        transport
            .hello_paths
            .borrow_mut()
            .extend(["a".to_string(), "b".to_string()]);
        transport
            .hello_uids
            .borrow_mut()
            .extend([("a".to_string(), Some(7)), ("b".to_string(), Some(7))]);
        transport
            .hello_capabilities
            .borrow_mut()
            .insert("b".to_string(), crate::packet::CAPABILITY_TIME_SYNC);
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);

        let results = manager.probe().unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.verified));
        assert_eq!(manager.verified_devices().len(), 1);
        assert_eq!(manager.verified_devices()[0].device_uid_hash, Some(7));
        assert_eq!(
            manager.verified_devices()[0].capabilities,
            crate::packet::CAPABILITY_TIME_SYNC
        );
    }

    #[test]
    fn ai_client_state_is_sent_only_to_capable_devices() {
        let capable = device_with_capabilities("a", CAPABILITY_AI_CLIENT_STATE);
        let unsupported = device("b");
        let transport = MockTransport {
            candidates: RefCell::new(vec![capable.clone(), unsupported.clone()]),
            ..MockTransport::default()
        };
        transport
            .hello_paths
            .borrow_mut()
            .extend(["a".to_string(), "b".to_string()]);
        transport
            .hello_capabilities
            .borrow_mut()
            .insert("a".to_string(), CAPABILITY_AI_CLIENT_STATE);
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let state = AiClientStatePacket::new(
            crate::packet::AiClientType::Codex,
            crate::packet::AiClientVariant::Cli as u8,
            true,
            crate::packet::AiActivityState::Available,
            crate::packet::AiWorkPhase::Unspecified,
            7,
        )
        .unwrap();

        assert_eq!(manager.send_ai_client_state(state, false).unwrap(), 1);

        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "a");
        assert_eq!(writes[0].1[4], crate::packet::PacketType::StateUpdate as u8);
    }

    #[test]
    fn ai_client_state_uses_capability_specific_payloads_and_phase_only_targeting() {
        let legacy_caps = CAPABILITY_AI_CLIENT_STATE;
        let detailed_caps = CAPABILITY_AI_CLIENT_STATE | CAPABILITY_AI_CLIENT_WORK_PHASE;
        let legacy = device_with_capabilities("legacy", legacy_caps);
        let detailed = device_with_capabilities("detailed", detailed_caps);
        let transport = MockTransport {
            candidates: RefCell::new(vec![legacy, detailed]),
            ..MockTransport::default()
        };
        transport
            .hello_paths
            .borrow_mut()
            .extend(["legacy".to_string(), "detailed".to_string()]);
        transport.hello_capabilities.borrow_mut().extend([
            ("legacy".to_string(), legacy_caps),
            ("detailed".to_string(), detailed_caps),
        ]);
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let state = AiClientStatePacket::new(
            crate::packet::AiClientType::Codex,
            crate::packet::AiClientVariant::Cli as u8,
            true,
            crate::packet::AiActivityState::Working,
            crate::packet::AiWorkPhase::Searching,
            11,
        )
        .unwrap();

        assert_eq!(manager.send_ai_client_state(state, false).unwrap(), 2);
        {
            let writes = manager.transport.writes.borrow();
            let legacy_write = writes.iter().find(|write| write.0 == "legacy").unwrap();
            let detailed_write = writes.iter().find(|write| write.0 == "detailed").unwrap();
            assert_eq!(legacy_write.1[9], 6);
            assert_eq!(detailed_write.1[9], 7);
            assert_eq!(
                detailed_write.1[19],
                crate::packet::AiWorkPhase::Searching as u8
            );
        }

        manager.transport.writes.borrow_mut().clear();
        assert_eq!(manager.send_ai_client_state(state, true).unwrap(), 1);
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "detailed");
        assert_eq!(writes[0].1[9], 7);
    }

    #[test]
    fn ai_client_state_sends_slot_payload_only_to_slot_capable_devices() {
        let legacy_caps = CAPABILITY_AI_CLIENT_STATE | CAPABILITY_AI_CLIENT_WORK_PHASE;
        let slot_caps = legacy_caps | CAPABILITY_AI_CLIENT_DISPLAY_SLOT;
        let legacy = device_with_capabilities("legacy", legacy_caps);
        let slot = device_with_capabilities("slot", slot_caps);
        let transport = MockTransport {
            candidates: RefCell::new(vec![legacy, slot]),
            ..MockTransport::default()
        };
        transport
            .hello_paths
            .borrow_mut()
            .extend(["legacy".to_string(), "slot".to_string()]);
        transport.hello_capabilities.borrow_mut().extend([
            ("legacy".to_string(), legacy_caps),
            ("slot".to_string(), slot_caps),
        ]);
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let state = AiClientStatePacket::new_for_slot(
            crate::packet::AiClientType::Codex,
            crate::packet::AiClientVariant::Cli as u8,
            true,
            crate::packet::AiActivityState::Working,
            crate::packet::AiWorkPhase::Executing,
            13,
            2,
        )
        .unwrap();

        assert_eq!(manager.send_ai_client_state(state, false).unwrap(), 1);
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "slot");
        assert_eq!(writes[0].1[9], 8);
        assert_eq!(writes[0].1[20], 2);
    }

    #[test]
    fn ai_client_state_sends_screenkey_payload_only_to_screenkey_capable_devices() {
        let slot_caps = CAPABILITY_AI_CLIENT_STATE
            | CAPABILITY_AI_CLIENT_WORK_PHASE
            | CAPABILITY_AI_CLIENT_DISPLAY_SLOT;
        let screenkey_caps = slot_caps | CAPABILITY_AI_CLIENT_SCREENKEY_STATE;
        // Distinct first bytes: `DeviceInfo::device_uid_hash` in this test
        // helper derives from the path's first byte, and two devices that
        // hash the same are treated as one physical keyboard.
        let slot_only = device_with_capabilities("basic", slot_caps);
        let screenkey = device_with_capabilities("keyed", screenkey_caps);
        let transport = MockTransport {
            candidates: RefCell::new(vec![slot_only, screenkey]),
            ..MockTransport::default()
        };
        transport
            .hello_paths
            .borrow_mut()
            .extend(["basic".to_string(), "keyed".to_string()]);
        transport.hello_capabilities.borrow_mut().extend([
            ("basic".to_string(), slot_caps),
            ("keyed".to_string(), screenkey_caps),
        ]);
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let state = AiClientStatePacket::new_for_slot(
            crate::packet::AiClientType::Codex,
            crate::packet::AiClientVariant::Cli as u8,
            true,
            crate::packet::AiActivityState::WaitingApproval,
            crate::packet::AiWorkPhase::Unspecified,
            21,
            1,
        )
        .unwrap()
        .with_screenkey_state(crate::packet::AiScreenKeyState::WaitingTarget);

        assert_eq!(manager.send_ai_client_state(state, false).unwrap(), 2);
        let writes = manager.transport.writes.borrow();
        let slot_write = writes.iter().find(|write| write.0 == "basic").unwrap();
        let screenkey_write = writes.iter().find(|write| write.0 == "keyed").unwrap();

        // A device without bit 14 must see byte-for-byte the same 8-byte
        // payload as before -- not one byte different.
        assert_eq!(slot_write.1[9], 8);
        assert_eq!(slot_write.1[20], 1);

        assert_eq!(screenkey_write.1[9], 9);
        assert_eq!(screenkey_write.1[20], 1);
        assert_eq!(
            screenkey_write.1[21],
            crate::packet::AiScreenKeyState::WaitingTarget as u8
        );
        assert_eq!(
            &screenkey_write.1[13..21],
            &slot_write.1[13..21],
            "bytes 0..8 of the screenkey payload must match the display-slot payload exactly"
        );
    }

    #[test]
    fn claude_code_state_is_sent_only_to_devices_advertising_its_renderer() {
        let legacy_caps = CAPABILITY_AI_CLIENT_STATE | CAPABILITY_AI_CLIENT_WORK_PHASE;
        let claude_caps = legacy_caps | CAPABILITY_AI_CLIENT_CLAUDE_CODE;
        let legacy = device_with_capabilities("legacy", legacy_caps);
        let claude = device_with_capabilities("claude", claude_caps);
        let transport = MockTransport {
            candidates: RefCell::new(vec![legacy, claude]),
            ..MockTransport::default()
        };
        transport
            .hello_paths
            .borrow_mut()
            .extend(["legacy".to_string(), "claude".to_string()]);
        transport.hello_capabilities.borrow_mut().extend([
            ("legacy".to_string(), legacy_caps),
            ("claude".to_string(), claude_caps),
        ]);
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let state = AiClientStatePacket::new(
            crate::packet::AiClientType::ClaudeCode,
            crate::packet::AiClientVariant::Cli as u8,
            true,
            crate::packet::AiActivityState::Available,
            crate::packet::AiWorkPhase::Unspecified,
            12,
        )
        .unwrap();

        assert_eq!(manager.send_ai_client_state(state, false).unwrap(), 1);
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "claude");
        assert_eq!(writes[0].1[13], 0x02);
    }

    #[test]
    fn probe_tolerates_transient_hello_misses_for_known_candidates() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let generation = manager.device_generation();

        manager.transport.hello_paths.borrow_mut().clear();

        manager.probe().unwrap();
        assert_eq!(manager.verified_devices(), &[device("a")]);
        assert_eq!(manager.device_generation(), generation);

        manager.probe().unwrap();
        assert_eq!(manager.verified_devices(), &[device("a")]);
        assert_eq!(manager.device_generation(), generation);

        manager.probe().unwrap();
        assert!(manager.verified_devices().is_empty());
        assert_ne!(manager.device_generation(), generation);
    }

    #[test]
    fn probe_removes_missing_candidate_immediately() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();

        manager.transport.candidates.borrow_mut().clear();
        manager.probe().unwrap();

        assert!(manager.verified_devices().is_empty());
    }

    #[test]
    fn sends_to_multiple_verified_devices() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a"), device("b")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        transport.hello_paths.borrow_mut().insert("b".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();

        assert_eq!(manager.send_set_layer(4).unwrap(), 2);
    }

    #[test]
    fn successful_write_does_not_change_device_generation() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let generation = manager.device_generation();

        assert_eq!(manager.send_set_layer(4).unwrap(), 1);

        assert_eq!(manager.device_generation(), generation);
    }

    #[test]
    fn write_failure_removes_device() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a"), device("b")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        transport.hello_paths.borrow_mut().insert("b".to_string());
        transport
            .failing_writes
            .borrow_mut()
            .insert("a".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();

        assert_eq!(manager.send_clear().unwrap(), 1);

        assert_eq!(manager.verified_devices(), &[device("b")]);
        assert_eq!(manager.device_generation(), 2);
    }
    #[test]
    fn drain_uplink_decodes_pending_packets_in_order() {
        use crate::packet::{BatteryEntry, BatteryStatusPacket, LayerStatePacket, UplinkPacket};

        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        let battery = UplinkPacket::Battery(BatteryStatusPacket {
            entries: vec![BatteryEntry {
                source: 0,
                level: Some(80),
            }],
        });
        let layer = UplinkPacket::LayerState(LayerStatePacket {
            active_layer: 2,
            layer_mask: 0,
            seq: 1,
        });
        transport.uplink.borrow_mut().insert(
            "a".to_string(),
            VecDeque::from(vec![battery.encode_payload(), layer.encode_payload()]),
        );
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();

        let events = manager.drain_uplink();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].1, battery);
        assert_eq!(events[1].1, layer);
        assert!(manager.drain_uplink().is_empty());
    }

    #[test]
    fn drain_uplink_skips_invalid_packets() {
        use crate::packet::{LayerStatePacket, UplinkPacket};

        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        let mut garbage = [0u8; PACKET_SIZE];
        garbage[0..2].copy_from_slice(b"HL");
        garbage[2] = 1;
        garbage[3] = 0x70;
        garbage[4] = 99; // invalid layer
        let valid = UplinkPacket::LayerState(LayerStatePacket {
            active_layer: 1,
            layer_mask: 0,
            seq: 0,
        });
        transport.uplink.borrow_mut().insert(
            "a".to_string(),
            VecDeque::from(vec![garbage, valid.encode_payload()]),
        );
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();

        let events = manager.drain_uplink();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].1, valid);
        assert_eq!(manager.verified_devices().len(), 1);
    }

    #[test]
    fn drain_uplink_read_error_removes_device() {
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a"), device("b")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        transport.hello_paths.borrow_mut().insert("b".to_string());
        transport.failing_reads.borrow_mut().insert("a".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe().unwrap();
        let generation = manager.device_generation();

        let events = manager.drain_uplink();

        assert!(events.is_empty());
        assert_eq!(manager.verified_devices(), &[device("b")]);
        assert_ne!(manager.device_generation(), generation);
    }

    #[test]
    fn config_get_info_does_not_send_without_capability() {
        let transport = MockTransport::default();
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device("a");

        let err = manager.config_get_encoder_info(&device).unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcUnsupported));
        assert!(manager.transport.writes.borrow().is_empty());
    }

    #[test]
    fn config_get_info_returns_matching_response() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::encoder_get_info_ok(
            0,
            crate::packet::EncoderGetInfo {
                layer_count: 4,
                encoder_count: 2,
                capabilities: 0,
                scroll_value: None,
                encoder_tap_ms: None,
            },
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let info = manager.config_get_encoder_info(&device).unwrap();

        assert_eq!(info.layer_count, 4);
        assert_eq!(info.encoder_count, 2);
        assert_eq!(manager.transport.writes.borrow().len(), 1);
    }

    #[test]
    fn config_get_info_ignores_uplink_and_mismatched_response() {
        let transport = MockTransport::default();
        let battery = crate::packet::UplinkPacket::Battery(crate::packet::BatteryStatusPacket {
            entries: vec![crate::packet::BatteryEntry {
                source: 0,
                level: Some(80),
            }],
        })
        .encode_payload();
        let stale = crate::packet::ConfigResponse::encoder_get_info_ok(
            99,
            crate::packet::EncoderGetInfo {
                layer_count: 1,
                encoder_count: 1,
                capabilities: 0,
                scroll_value: None,
                encoder_tap_ms: None,
            },
        )
        .encode_payload();
        let matching = crate::packet::ConfigResponse::encoder_get_info_ok(
            0,
            crate::packet::EncoderGetInfo {
                layer_count: 5,
                encoder_count: 2,
                capabilities: 0,
                scroll_value: None,
                encoder_tap_ms: None,
            },
        )
        .encode_payload();
        transport.uplink.borrow_mut().insert(
            "a".to_string(),
            VecDeque::from(vec![battery, stale, matching]),
        );
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let info = manager.config_get_encoder_info(&device).unwrap();

        assert_eq!(info.layer_count, 5);
        assert_eq!(info.encoder_count, 2);
        assert_eq!(manager.transport.writes.borrow().len(), 1);
        let deferred = manager.drain_uplink();
        assert_eq!(deferred.len(), 1);
        assert!(matches!(deferred[0].1, UplinkPacket::Battery(_)));
    }

    #[test]
    fn config_get_info_retries_same_seq_once() {
        let transport = MockTransport::default();
        let config = HidConfig {
            hello_timeout_ms: 1,
            ..HidConfig::default()
        };
        let mut manager = HidDeviceManager::new(config, transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let err = manager.config_get_encoder_info(&device).unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcTimeout));
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].1, writes[1].1);
    }

    #[test]
    fn config_get_bindings_does_not_send_without_capability() {
        let transport = MockTransport::default();
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device("a");

        let err = manager
            .config_get_encoder_bindings(&device, 0, 0)
            .unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcUnsupported));
        assert!(manager.transport.writes.borrow().is_empty());
    }

    #[test]
    fn config_get_bindings_returns_matching_response() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::encoder_get_bindings_ok(
            0,
            crate::packet::EncoderGetBindings {
                layer_id: 3,
                encoder_id: 1,
                source: crate::packet::EncoderBindingSource::Keymap,
                flags: crate::packet::EncoderBindingFlags::default(),
                cw_binding: crate::packet::EncoderBinding {
                    behavior_id: 0,
                    param1: 0,
                    param2: 0,
                },
                ccw_binding: crate::packet::EncoderBinding {
                    behavior_id: 0,
                    param1: 0,
                    param2: 0,
                },
            },
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let bindings = manager.config_get_encoder_bindings(&device, 3, 1).unwrap();

        assert_eq!(bindings.layer_id, 3);
        assert_eq!(bindings.encoder_id, 1);
        assert_eq!(bindings.source, crate::packet::EncoderBindingSource::Keymap);
        assert_eq!(bindings.flags.bits(), 0);
        assert_eq!(manager.transport.writes.borrow().len(), 1);
    }

    #[test]
    fn config_get_bindings_ignores_uplink_and_mismatched_response() {
        let transport = MockTransport::default();
        let battery = crate::packet::UplinkPacket::Battery(crate::packet::BatteryStatusPacket {
            entries: vec![crate::packet::BatteryEntry {
                source: 0,
                level: Some(80),
            }],
        })
        .encode_payload();
        let stale = crate::packet::ConfigResponse::encoder_get_bindings_ok(
            99,
            crate::packet::EncoderGetBindings {
                layer_id: 0,
                encoder_id: 0,
                source: crate::packet::EncoderBindingSource::Keymap,
                flags: crate::packet::EncoderBindingFlags::default(),
                cw_binding: crate::packet::EncoderBinding {
                    behavior_id: 0,
                    param1: 0,
                    param2: 0,
                },
                ccw_binding: crate::packet::EncoderBinding {
                    behavior_id: 0,
                    param1: 0,
                    param2: 0,
                },
            },
        )
        .encode_payload();
        let matching = crate::packet::ConfigResponse::encoder_get_bindings_ok(
            0,
            crate::packet::EncoderGetBindings {
                layer_id: 5,
                encoder_id: 1,
                source: crate::packet::EncoderBindingSource::Keymap,
                flags: crate::packet::EncoderBindingFlags::default(),
                cw_binding: crate::packet::EncoderBinding {
                    behavior_id: 0,
                    param1: 0,
                    param2: 0,
                },
                ccw_binding: crate::packet::EncoderBinding {
                    behavior_id: 0,
                    param1: 0,
                    param2: 0,
                },
            },
        )
        .encode_payload();
        transport.uplink.borrow_mut().insert(
            "a".to_string(),
            VecDeque::from(vec![battery, stale, matching]),
        );
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let bindings = manager.config_get_encoder_bindings(&device, 5, 1).unwrap();

        assert_eq!(bindings.layer_id, 5);
        assert_eq!(bindings.encoder_id, 1);
        assert_eq!(manager.transport.writes.borrow().len(), 1);
    }

    #[test]
    fn config_get_bindings_returns_config_status_error() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::status(
            0,
            crate::packet::ConfigFeature::Encoder as u8,
            crate::packet::ConfigOp::GetBindings as u8,
            crate::packet::ConfigStatus::InvalidArgument,
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let err = manager
            .config_get_encoder_bindings(&device, 9, 2)
            .unwrap_err();

        assert!(matches!(
            err,
            HidError::ConfigRpcStatus(crate::packet::ConfigStatus::InvalidArgument)
        ));
    }

    #[test]
    fn config_get_bindings_retries_same_seq_once() {
        let transport = MockTransport::default();
        let config = HidConfig {
            hello_timeout_ms: 1,
            ..HidConfig::default()
        };
        let mut manager = HidDeviceManager::new(config, transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let err = manager
            .config_get_encoder_bindings(&device, 0, 0)
            .unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcTimeout));
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].1, writes[1].1);
    }

    /// Two manager instances start their seq at 0, so a GET_BINDINGS reply for
    /// another encoder can share (seq, feature, op) with our request. The OK
    /// response must additionally echo the requested layer/encoder target.
    #[test]
    fn config_get_bindings_ignores_mismatched_target_echo() {
        let zero_binding = crate::packet::EncoderBinding {
            behavior_id: 0,
            param1: 0,
            param2: 0,
        };
        let transport = MockTransport::default();
        let wrong_target = crate::packet::ConfigResponse::encoder_get_bindings_ok(
            0,
            crate::packet::EncoderGetBindings {
                layer_id: 5,
                encoder_id: 0,
                source: crate::packet::EncoderBindingSource::Keymap,
                flags: crate::packet::EncoderBindingFlags::default(),
                cw_binding: zero_binding,
                ccw_binding: zero_binding,
            },
        )
        .encode_payload();
        let matching = crate::packet::ConfigResponse::encoder_get_bindings_ok(
            0,
            crate::packet::EncoderGetBindings {
                layer_id: 5,
                encoder_id: 1,
                source: crate::packet::EncoderBindingSource::Keymap,
                flags: crate::packet::EncoderBindingFlags::default(),
                cw_binding: zero_binding,
                ccw_binding: zero_binding,
            },
        )
        .encode_payload();
        let transport_uplink = vec![wrong_target, matching];
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(transport_uplink));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let bindings = manager.config_get_encoder_bindings(&device, 5, 1).unwrap();

        assert_eq!(bindings.layer_id, 5);
        assert_eq!(bindings.encoder_id, 1);
    }

    #[test]
    fn config_set_bindings_does_not_send_without_capability() {
        let transport = MockTransport::default();
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device("a");

        let err = manager
            .config_set_encoder_bindings(
                &device,
                0,
                0,
                crate::packet::EncoderBinding {
                    behavior_id: 1,
                    param1: 2,
                    param2: 3,
                },
                crate::packet::EncoderBinding {
                    behavior_id: 4,
                    param1: 5,
                    param2: 6,
                },
            )
            .unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcUnsupported));
        assert!(manager.transport.writes.borrow().is_empty());
    }

    #[test]
    fn config_set_bindings_sends_request_and_returns_ok() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::status(
            0,
            crate::packet::ConfigFeature::Encoder as u8,
            crate::packet::ConfigOp::SetBindings as u8,
            crate::packet::ConfigStatus::Ok,
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        manager
            .config_set_encoder_bindings(
                &device,
                0x01020304,
                2,
                crate::packet::EncoderBinding {
                    behavior_id: 0x1234,
                    param1: 56,
                    param2: 78,
                },
                crate::packet::EncoderBinding {
                    behavior_id: 0x5678,
                    param1: 90,
                    param2: 12,
                },
            )
            .unwrap();

        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        let report = writes[0].1;
        let payload = &report[1..];
        assert_eq!(payload[3], crate::packet::PacketType::ConfigRequest as u8);
        assert_eq!(payload[5], crate::packet::ConfigFeature::Encoder as u8);
        assert_eq!(payload[6], crate::packet::ConfigOp::SetBindings as u8);
        assert_eq!(payload[8], 28);
        assert_eq!(
            &payload[crate::packet::PAYLOAD_OFFSET..crate::packet::PAYLOAD_OFFSET + 4],
            &[4, 3, 2, 1]
        );
        assert_eq!(payload[crate::packet::PAYLOAD_OFFSET + 4], 2);
        assert_eq!(payload[crate::packet::PAYLOAD_OFFSET + 5], 0x03);
        assert_eq!(payload[crate::packet::PAYLOAD_OFFSET + 6], 0);
        assert_eq!(payload[crate::packet::PAYLOAD_OFFSET + 7], 0);
    }

    #[test]
    fn config_set_bindings_returns_config_status_error() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::status(
            0,
            crate::packet::ConfigFeature::Encoder as u8,
            crate::packet::ConfigOp::SetBindings as u8,
            crate::packet::ConfigStatus::InvalidArgument,
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let err = manager
            .config_set_encoder_bindings(
                &device,
                9,
                2,
                crate::packet::EncoderBinding {
                    behavior_id: 1,
                    param1: 2,
                    param2: 3,
                },
                crate::packet::EncoderBinding {
                    behavior_id: 4,
                    param1: 5,
                    param2: 6,
                },
            )
            .unwrap_err();

        assert!(matches!(
            err,
            HidError::ConfigRpcStatus(crate::packet::ConfigStatus::InvalidArgument)
        ));
    }

    #[test]
    fn config_set_bindings_retries_same_seq_once() {
        let transport = MockTransport::default();
        let config = HidConfig {
            hello_timeout_ms: 1,
            ..HidConfig::default()
        };
        let mut manager = HidDeviceManager::new(config, transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let err = manager
            .config_set_encoder_bindings(
                &device,
                0,
                0,
                crate::packet::EncoderBinding {
                    behavior_id: 1,
                    param1: 2,
                    param2: 3,
                },
                crate::packet::EncoderBinding {
                    behavior_id: 4,
                    param1: 5,
                    param2: 6,
                },
            )
            .unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcTimeout));
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].1, writes[1].1);
    }

    #[test]
    fn config_get_dirty_does_not_send_without_capability() {
        let transport = MockTransport::default();
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device("a");

        let err = manager.config_get_encoder_dirty(&device).unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcUnsupported));
        assert!(manager.transport.writes.borrow().is_empty());
    }

    #[test]
    fn config_get_dirty_returns_matching_response() {
        let transport = MockTransport::default();
        let response =
            crate::packet::ConfigResponse::encoder_get_dirty_ok(0, true).encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let dirty = manager.config_get_encoder_dirty(&device).unwrap();

        assert!(dirty);
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        let payload = &writes[0].1[1..];
        assert_eq!(payload[3], crate::packet::PacketType::ConfigRequest as u8);
        assert_eq!(payload[5], crate::packet::ConfigFeature::Encoder as u8);
        assert_eq!(payload[6], crate::packet::ConfigOp::GetDirty as u8);
        assert_eq!(payload[8], 0);
    }

    #[test]
    fn config_save_encoder_sends_request_and_returns_ok() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::status(
            0,
            crate::packet::ConfigFeature::Encoder as u8,
            crate::packet::ConfigOp::Save as u8,
            crate::packet::ConfigStatus::Ok,
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        manager.config_save_encoder(&device).unwrap();

        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        let payload = &writes[0].1[1..];
        assert_eq!(payload[3], crate::packet::PacketType::ConfigRequest as u8);
        assert_eq!(payload[5], crate::packet::ConfigFeature::Encoder as u8);
        assert_eq!(payload[6], crate::packet::ConfigOp::Save as u8);
        assert_eq!(payload[8], 0);
    }

    #[test]
    fn config_discard_encoder_returns_config_status_error() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::status(
            0,
            crate::packet::ConfigFeature::Encoder as u8,
            crate::packet::ConfigOp::Discard as u8,
            crate::packet::ConfigStatus::StorageError,
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let err = manager.config_discard_encoder(&device).unwrap_err();

        assert!(matches!(
            err,
            HidError::ConfigRpcStatus(crate::packet::ConfigStatus::StorageError)
        ));
    }

    #[test]
    fn config_clear_encoder_override_sends_payload_and_returns_ok() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::status(
            0,
            crate::packet::ConfigFeature::Encoder as u8,
            crate::packet::ConfigOp::ClearOverride as u8,
            crate::packet::ConfigStatus::Ok,
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        manager
            .config_clear_encoder_override(&device, 0x01020304, 7)
            .unwrap();

        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 1);
        let payload = &writes[0].1[1..];
        assert_eq!(payload[3], crate::packet::PacketType::ConfigRequest as u8);
        assert_eq!(payload[5], crate::packet::ConfigFeature::Encoder as u8);
        assert_eq!(payload[6], crate::packet::ConfigOp::ClearOverride as u8);
        assert_eq!(payload[8], 5);
        assert_eq!(
            &payload[crate::packet::PAYLOAD_OFFSET..crate::packet::PAYLOAD_OFFSET + 4],
            &[4, 3, 2, 1]
        );
        assert_eq!(payload[crate::packet::PAYLOAD_OFFSET + 4], 7);
    }

    #[test]
    fn config_get_dirty_retries_same_seq_once() {
        let transport = MockTransport::default();
        let config = HidConfig {
            hello_timeout_ms: 1,
            ..HidConfig::default()
        };
        let mut manager = HidDeviceManager::new(config, transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let err = manager.config_get_encoder_dirty(&device).unwrap_err();

        assert!(matches!(err, HidError::ConfigRpcTimeout));
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].1, writes[1].1);
    }

    fn combo_item(slot: u8) -> crate::packet::ComboItem {
        crate::packet::ComboItem::new(
            slot,
            "test",
            &[1, 2],
            false,
            crate::packet::ComboBinding {
                behavior_id: 1,
                param1: 2,
                param2: 3,
            },
            0,
            50,
            None,
        )
        .unwrap()
    }

    #[test]
    fn combo_get_info_ignores_mismatched_seq_feature_and_op() {
        let info = crate::packet::ComboInfo {
            max_combos: 32,
            max_keys_per_combo: 8,
            combo_count: 6,
            flags: crate::packet::ComboInfoFlags::default(),
            occupied_slots: 0x3f,
            stale_slots: 0,
            invalid_slots: 0,
        };
        let transport = MockTransport::default();
        let responses = vec![
            crate::packet::ConfigResponse::combo_get_info_ok(99, info).encode_payload(),
            crate::packet::ConfigResponse::status(
                0,
                crate::packet::ConfigFeature::Encoder as u8,
                crate::packet::ConfigOp::GetInfo as u8,
                crate::packet::ConfigStatus::Ok,
            )
            .encode_payload(),
            crate::packet::ConfigResponse::status(
                0,
                crate::packet::ConfigFeature::Combo as u8,
                crate::packet::ComboConfigOp::Save as u8,
                crate::packet::ConfigStatus::Ok,
            )
            .encode_payload(),
            crate::packet::ConfigResponse::combo_get_info_ok(0, info).encode_payload(),
        ];
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(responses));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        assert_eq!(manager.config_get_combo_info(&device).unwrap(), info);
        assert_eq!(manager.transport.writes.borrow().len(), 1);
    }

    #[test]
    fn combo_get_info_retries_identical_request_once_on_timeout() {
        let transport = MockTransport::default();
        let config = HidConfig {
            hello_timeout_ms: 1,
            ..HidConfig::default()
        };
        let mut manager = HidDeviceManager::new(config, transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        let error = manager.config_get_combo_info(&device).unwrap_err();

        assert!(matches!(error, HidError::ConfigRpcTimeout));
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].1, writes[1].1);
    }

    #[test]
    fn combo_get_combo_maps_not_found_status() {
        let transport = MockTransport::default();
        let response = crate::packet::ConfigResponse::status(
            0,
            crate::packet::ConfigFeature::Combo as u8,
            crate::packet::ComboConfigOp::GetCombo as u8,
            crate::packet::ConfigStatus::NotFound,
        )
        .encode_payload();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(vec![response]));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        assert!(matches!(
            manager.config_get_combo(&device, 31).unwrap_err(),
            HidError::ConfigRpcStatus(crate::packet::ConfigStatus::NotFound)
        ));
    }

    #[test]
    fn combo_set_maps_busy_and_save_maps_storage_error() {
        let transport = MockTransport::default();
        let responses = vec![
            crate::packet::ConfigResponse::status(
                0,
                crate::packet::ConfigFeature::Combo as u8,
                crate::packet::ComboConfigOp::SetCombo as u8,
                crate::packet::ConfigStatus::Busy,
            )
            .encode_payload(),
            crate::packet::ConfigResponse::status(
                1,
                crate::packet::ConfigFeature::Combo as u8,
                crate::packet::ComboConfigOp::Save as u8,
                crate::packet::ConfigStatus::StorageError,
            )
            .encode_payload(),
        ];
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(responses));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        assert!(matches!(
            manager
                .config_set_combo(&device, combo_item(31))
                .unwrap_err(),
            HidError::ConfigRpcStatus(crate::packet::ConfigStatus::Busy)
        ));
        assert!(matches!(
            manager.config_save_combos(&device).unwrap_err(),
            HidError::ConfigRpcStatus(crate::packet::ConfigStatus::StorageError)
        ));
    }

    #[test]
    fn combo_delete_discard_and_reset_send_typed_ops() {
        let transport = MockTransport::default();
        let responses = [
            crate::packet::ComboConfigOp::DeleteCombo,
            crate::packet::ComboConfigOp::Discard,
            crate::packet::ComboConfigOp::ResetToKeymap,
        ]
        .into_iter()
        .enumerate()
        .map(|(seq, op)| {
            crate::packet::ConfigResponse::status(
                seq as u8,
                crate::packet::ConfigFeature::Combo as u8,
                op as u8,
                crate::packet::ConfigStatus::Ok,
            )
            .encode_payload()
        })
        .collect::<Vec<_>>();
        transport
            .uplink
            .borrow_mut()
            .insert("a".to_string(), VecDeque::from(responses));
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        let device = device_with_capabilities("a", crate::packet::CAPABILITY_CONFIG_RPC);

        manager.config_delete_combo(&device, 31).unwrap();
        manager.config_discard_combos(&device).unwrap();
        manager.config_reset_combos_to_keymap(&device).unwrap();
        let writes = manager.transport.writes.borrow();
        assert_eq!(writes.len(), 3);
        assert_eq!(writes[0].1[1 + crate::packet::PAYLOAD_OFFSET], 31);
    }

    #[test]
    fn periodic_rescan_adds_new_verified_devices() {
        let start = Instant::now();
        let transport = MockTransport {
            candidates: RefCell::new(vec![device("a")]),
            ..MockTransport::default()
        };
        transport.hello_paths.borrow_mut().insert("a".to_string());
        let mut manager = HidDeviceManager::new(HidConfig::default(), transport);
        manager.probe_at(start).unwrap();
        let generation = manager.device_generation();

        manager.transport.candidates.borrow_mut().push(device("b"));
        manager
            .transport
            .hello_paths
            .borrow_mut()
            .insert("b".to_string());

        manager
            .ensure_verified_at(start + Duration::from_secs(4))
            .unwrap();
        assert_eq!(manager.verified_devices(), &[device("a")]);
        assert_eq!(manager.device_generation(), generation);

        manager
            .ensure_verified_at(start + Duration::from_secs(5))
            .unwrap();
        assert_eq!(manager.verified_devices(), &[device("a"), device("b")]);
        assert_ne!(manager.device_generation(), generation);
    }
}
