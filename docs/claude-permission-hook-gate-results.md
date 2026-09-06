# Claude Code PermissionRequest hook Gate (KO-3) 結果

- 状態: 完了
- 実施日: 2026-09-03
- 最終判定: **KO-3 成立**（キーボードから Claude Code の承認・拒否を注入できる）
- 追記: 2026-09-06 に Q5 を追試し、**成立**を確認した。§5 の「常に許可を実装しない」判断は
  撤回されている。詳細は §5 冒頭と `ai-approval-hud-design.md` §9.3
- 対象 Claude Code: `2.1.259`
- 実行環境: Windows 11 Pro `10.0.26200.9278`、Windows PowerShell 5.1
- 検証コード: `tools/keylink-claude-permission-probe.ps1`
- 検証方法: Keylink Studio を一切経由せず、独立したローカルHTTPサーバを
  `PermissionRequest` hook の宛先にして実機の Claude Code と対話した

---

## 1. 背景

`hud-focus-gate-results.md` で HUD（モニタ上の常時最前面パネル）がフォーカスを奪わないことが
確定した。次の問いは「その HUD で読んだ内容に対する回答を、実際に AI クライアントへ注入できるか」である。

Codex は Broker 経由で JSON-RPC response を代理送信できる見込みがあるが（KO-2 で検証）、
**Claude Code には正規の回答APIが存在しない。** 唯一の可能性が `PermissionRequest` hook から
decision を返す経路であり、これが不成立なら Claude Code は「HUD で内容は見えるが回答はターミナル」
へ縮退する。

Keylink Studio の現状は次のとおりである。

- `crates/rawhid-host-core/src/claude_hooks.rs` の `hooks_json()` が hook を登録する。
  `PermissionRequest` は **`type: "http"`**、`matcher: "*"`、`timeout` は
  `CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS = 1`（秒）。`type: "command"` は `SessionStart` のみ
- `crates/rawhid-host-core/src/claude_observer.rs:323` が受信した全 hook へ **204 No Content** を返す。
  すなわち decision を返さない観測専用である

**Studio を改造してから測ると、失敗したときに「プロトコルが対応していない」のか
「Studioの実装が悪い」のか切り分けられない。** そのため本Gateでは Studio に一切手を入れず、
プロトコルの能力だけを独立に確定させた。

---

## 2. 判定

| # | 問い | 結果 |
|---|---|---|
| Q1 | hook をブロックしている間、ターミナルに何が表示されるか | **約3秒で通常の許可プロンプトが出る**（hook の応答を待たない） |
| Q2 | hook の応答を待つか | 待たないが、**後から返した decision が有効**（Q3で確認） |
| **Q3** | `decision: {behavior:"allow"}` は受理されるか | ✅ **成立**。ユーザーが一切操作せずに実行が進んだ |
| **Q4** | `decision: {behavior:"deny", message}` は受理されるか | ✅ **成立**。拒否され、`message` がモデルに届いた |
| **Q5** | `updatedPermissions` で恒久ルールを作れるか | ✅ **成立**（2026-09-06 に追試）。§5 の不採用判断は撤回された |
| **Q6** | サーバ不在・応答不能のとき | ✅ **固まらない。即座に通常プロンプトへ縮退** |

**結論: キーボードから Claude Code の承認・拒否を注入できる。**

---

## 3. 観測された挙動の詳細

### 3.1 Q1 / Q2 — 回答経路は2本ある

`PermissionRequest` hook を15秒ブロックしても、Claude Code は待たずに**約3秒でターミナルへ
通常の許可プロンプトを表示した**。ただしツールは実行されず、誰かが答えるまで停止している。

そして Q3 のとおり、**プロンプトが出たままの状態で hook が decision を返すと、それが採用される**。

つまり回答経路は次の2本が並行して存在する。

```text
        ┌─ ターミナルのプロンプト（人が答える）
要求 ───┤                                        → 先に答えたほうが1回だけ採用される
        └─ hook の decision（HUD経由で答える）
```

当初は「hook がブロックして画面には何も出ない」形を想定していたが、**実際の挙動のほうが望ましい**。

| | 想定（hookがブロック・画面に何も出ない） | 実際（並行） |
|---|---|---|
| Studio 停止時 | Claude Code が timeout まで固まる | **ターミナルで普通に答えられる** |
| フォールバック | 別途設計が必要 | **自動的に常に存在する** |

### 3.2 Q4 — 拒否理由がモデルに届く

`{"behavior":"deny","message":"KO-3 probe denied this request."}` を返したときの実機の表示。

```text
  Ran 1 shell command
Denied by PermissionRequest hook

New-Item -ItemType Directory -Path "re-ko3-test4" を実行しようとしましたが、
「KO-3 probe denied this request.」というエラーで拒否されています。
```

- ターミナルが **`Denied by PermissionRequest hook` と明示**する。ユーザーが理由を誤解しない
- **`message` の文字列がモデルまで届き、Claude が応答で引用した**

実装では「ユーザーがキーボードで拒否しました」「代わりに○○してください」といった指示を
添えられる。単なる拒否より使える。

### 3.3 Q6 — サーバ不在時

プローブを起動せずに承認が必要な操作を依頼したところ、**待たされることなく即座に通常の
プロンプトが表示された**。hook の `timeout` を長く設定しても、Studio 停止時にユーザーが
固まされるリスクはない。

---

## 4. hook body の実測内容

`PermissionRequest` の受信内容（実機）。

```json
{
  "session_id": "c21f2516-...",
  "transcript_path": "C:\\Users\\...\\<session>.jsonl",
  "cwd": "C:\\Users\\...\\keylink-claude-permission-probe-...",
  "scratchpad_dir": "C:\\Users\\...\\scratchpad",
  "prompt_id": "ad630989-...",
  "permission_mode": "acceptEdits",
  "effort": { "level": "high" },
  "hook_event_name": "PermissionRequest",
  "tool_name": "PowerShell",
  "tool_input": {
    "command": "New-Item -ItemType Directory ko3-test8",
    "description": "Create ko3-test8 directory"
  },
  "permission_suggestions": [
    {
      "type": "addRules",
      "rules": [
        { "toolName": "PowerShell", "ruleContent": "New-Item -ItemType Directory ko3-test8" }
      ],
      "behavior": "allow",
      "destination": "localSettings"
    }
  ]
}
```

HTTPリクエストは `Content-Type: application/json`、`User-Agent: axios/1.15.2`、`Connection: keep-alive`。

HUDへの表示に必要な情報はすべて揃っている。

| HUDに出す内容 | 取得元 |
|---|---|
| 何を実行しようとしているか | `tool_input.command`（全文） |
| その説明 | `tool_input.description` |
| ツール種別 | `tool_name` |
| 作業ディレクトリ | `cwd` |
| どのセッションか | `session_id` |

補足:

- この環境ではシェル実行の `tool_name` は `Bash` ではなく **`PowerShell`**
- `suppress_always_allow_rule` は観測した全ケースで**存在しなかった**（`(absent)`）。
  存在しうるフィールドとして扱い、あれば「常に許可」を出さないこと
- **画面に出ている選択肢のリストは hook body に含まれない。** HUD は Host 側で
  `許可 / 拒否` を正規化して提示することになる（HUDが正本なので序数一致の問題は発生しない）

---

## 5. Q5（常に許可）を不採用とする根拠

> **2026-09-06 追記: この節の結論は撤回された。** 下記の根拠は 2026-08-08 の1件の観測に
> もとづいており、追試で否定された。実際には `destination: "session"`（セッション限定）や
> `//c/temp/**` のようなワイルドカードのルールも来る。`updatedPermissions` に候補をそのまま
> 載せて返せば適用されることも実測で確認した（`localSettings` 宛では実際に
> `.claude/settings.local.json` が作成された）。現在の方針は
> `ai-approval-hud-design.md` §9.3 が正本で、**Claude Code でも「常に許可」を提供する**。
> ただし恒久宛の候補も現役で存在するため、**適用範囲を HUD のラベルに必ず出す**。
> 以下は撤回された当時の判断として残す。

検証は完了していないが、**実装しない**と判断した。理由は検証の困難さではなく、
検証の過程で判明した次の3点である。

1. **`destination: "localSettings"` は恒久権限である。**
   成功すれば `.claude/settings.local.json` へ書かれ、次回起動以降もずっと効く。
   キーボードの1押しで恒久的な実行権限を作らせるのは重すぎる
2. **`ruleContent` は完全一致の文字列である。**
   実機では、同じ「`mkdir ko3-final` を実行して」という依頼に対して Claude が
   1回目と2回目で違うコマンド（2回目は `if (Test-Path ...) {...} else {...}`）を組み立てた。
   完全一致ルールは**次に同じ文字列が来る保証がなく、実効性が乏しい**
3. **ターミナルの選択肢のほうが広い。**
   同じ場面で TUI が提示していたのは `Yes, and don't ask again for: New-Item *`
   （ワイルドカード）であり、hook が渡してくる提案より広い。
   **キーボードから押した場合とターミナルで選んだ場合で、作られる権限の広さが変わる。**
   ユーザーからは同じ操作に見えるため、混乱を招く

加えて、複合コマンド（`if/else` を含むもの）では `permission_suggestions` 自体が届かず、
TUI 側にも「常に許可」の選択肢が出なかった。適用場面がそもそも狭い。

**方針: キーボードからは「許可 / 拒否」のみを提供する。**「常に許可」が必要なときは
ターミナルで選択肢を選んでもらう。そちらのほうが広く適切なルールが作られる。

---

## 6. Keylink Studio 側の変更点

決定を返すために必要な変更は **`claude_observer.rs:323` の1箇所**である。

```rust
let _ = write_response(&mut stream, 204).await;
```

ここで 204 の代わりに decision を含む JSON ボディを返す。応答を保留すれば、
その間 hook はブロックされる。

> 当初「`keylink-claude-hook` を観測専用からブロックする意思決定者へ構造反転させる必要がある」と
> 見積もっていたが、**これは誤りだった。** `PermissionRequest` は `type: "http"` hook であり
> helper プロセスを経由しない。helper が使われるのは `SessionStart` と wrapper 終了通知のみである。

実装時に決めること:

- `PermissionRequest` の応答だけを分岐させ、他の hook は 204 のまま維持する
- hook の `timeout` を 1秒から延長する（人が HUD を読んで判断する時間）。
  Q6 のとおり延長してもユーザーが固まるリスクはない
- **ターミナルと HUD の二重回答の調停。** 先に確定したほう1件だけを採用する
- Studio 側が回答を持たない場合は必ず 204 へ縮退させ、ターミナルに委ねる

---

## 7. 検証手段側で潰した欠陥

KO-1 と同様、**判定に至るまでの障害はすべて計測側にあった**。再発しやすいものを記録する。

### 7.1 `HttpListener.GetContext()` は Ctrl+C を受け付けない

同期ブロッキング呼び出しのため、PowerShell が停止要求を処理する隙がなく、`try/finally` の
`finally` にも到達しない。実機でターミナルを閉じるしか終了手段がなくなった。

`GetContextAsync()` ＋ `WaitOne(200)` のポーリングに変更して解決した。PowerShell は
**文の境界で**停止要求を処理するため、短い待機を挟むループにすれば Ctrl+C が効く。
あわせて `-MaxRequests` を追加し、1件測って自動終了できるようにした。

### 7.2 PowerShell は要素1個の配列をスカラーに畳む

`ConvertFrom-Json` → `ConvertTo-Json` の往復で、`permission_suggestions`（要素1個の配列）が
オブジェクトになっていた。そのため `updatedPermissions` に配列ではなくオブジェクトを送っており、
Q5 の最初の試行は**検証になっていなかった**。

受信した生テキストから該当フィールドの文字列範囲を切り出して、そのまま埋め込む方式へ変更した
（括弧の対応を数え、文字列リテラル内とエスケープを考慮する）。

**JSONを素通しする用途で PowerShell のパーサ／シリアライザを経由させてはいけない。**

### 7.3 `[string]` 型注釈は `$null` を空文字列に変える

`param([string]$X)` に `$null` を渡すと暗黙に `''` へ変換され、`$null -ne $X` が常に真になる。
「フィールドが見つからなかった」判定が壊れていた。無型にし `[string]::IsNullOrEmpty()` で判定する。

### 7.4 ログを起動のたびに削除してはいけない

`-MaxRequests 1` で1件ずつ測る運用では、1つの調査が「同じディレクトリに対してモードを変えて
複数回起動する」形になる。起動時にログを削除していたため、**AllowAlways の要求と応答の記録を
実際に失い、検証をやり直す羽目になった**。追記式に変更し、実行ごとの区切り行を入れた。

### 7.5 テスト設計上の落とし穴

- **Claude は同じ依頼に対して毎回同じコマンドを生成しない。**
  完全一致ルールの検証には、同一コマンドが2回出ることが前提になるが、これを安定して
  再現させるのが難しい
- **`Get-Date` のような無害なコマンドは承認を要求しない**ため `PermissionRequest` が発火せず、
  プローブが永久に待機する。検証には**確実に承認を要求する操作**を使うこと

---

## 8. 設計への影響

1. **Claude Code も HUD からの回答対象になる。** Codex と同じ体験を提供できる
2. **HUD は必須ではなく便利層である。** Studio が落ちていてもターミナルで普通に答えられる。
   フォールバックの設計コストがゼロになった
3. **拒否には理由を添えられる。** モデルに届くので、単に止めるだけでなく方向転換を指示できる
4. **キーボードからの「常に許可」は提供しない**（§5）
5. **二重回答の調停が必要。** ターミナルと HUD の両方が生きているため、先に確定したほうを
   1回だけ採用する排他制御を実装する

---

## 9. 非対象

- `PermissionRequest` 以外の hook から decision を返せるか
  （`PreToolUse` / `Stop` / `UserPromptSubmit` などは未検証）
- `Elicitation` hook からの回答注入
- `AskUserQuestion` の選択肢への回答
- hook `timeout` の上限値
- 二重回答が同時に発生したときの実機挙動

---

## 10. 次のノックアウト要因

| # | 内容 | 落ちた場合 |
|---|---|---|
| KO-2 | Codex へ代理 response を送ったとき TUI のプロンプトが正しく閉じるか | Broker が要求を保持して CLI へ転送しない方式へ切り替える |

---

## 11. 追記（2026-09-07 実機）— ターミナルが先に答えたときの hook

段階4 の実機確認で、**first-wins のターミナル側について当初の前提が違っていた**ことが分かった。

**ターミナルで拒否しても、Studio が待ち受けている hook は 1 つも届かない。** 拒否後 6 分間、
`PermissionDenied` も `PostToolUse` も `Stop` も来なかった。`PermissionDenied` は plugin に
登録済みだが、診断ログを入れて以降**一度も観測されていない**。

代わりに **hook 接続が閉じる**。これは対照実験で確認した。

| ターミナルでの操作 | `PermissionRequest` の hook 接続 | そのあと届く hook |
|---|---|---|
| **拒否** | **要求の 10.7 秒後に閉じた**（＝押した瞬間） | 無し |
| **許可** | 開いたまま | 道具の完了後に `Stop` / `PostToolUse` / `PostToolBatch` |
| **放置**（何も触らない） | **3 分半後も開いたまま** | `Notification` のみ |

放置した要求の接続が開いたままであることが、「10 秒ほどで勝手に閉じる作り」ではないことの
裏付けになっている。**接続の切断は「ターミナルで拒否された」合図として使える**（設計は
`docs/ai-approval-hud-design.md` §9.5）。

### 観測された hook の顔ぶれ

診断ログ（`claude hook observed`）で数えた実測。**`PermissionDenied` はゼロ**、
`PostToolUse` より `PostToolBatch` の方が多い。

| hook | 回数 |
|---|---|
| Notification | 10 |
| UserPromptSubmit / PreToolUse | 9 / 9 |
| PermissionRequest | 8 |
| Stop / SessionEnd / PostToolBatch | 2 / 2 / 2 |
| PostToolUse | 1 |
| **PermissionDenied** | **0** |

### §Q6 の再確認

hook の `timeout` を延ばして安全という §Q6 の結論にもとづき、2026-09-07 に
`PermissionRequest` の `timeout` を 60 秒 → 600 秒、Host 側の待ちを 55 秒 → 595 秒へ延ばした。
**Claude Code 側の上限は依然として未実測**である。
