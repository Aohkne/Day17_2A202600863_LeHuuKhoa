# Phân tích kết quả benchmark

Số liệu dưới đây lấy từ `python3 src/benchmark.py` (offline mode, deterministic), chạy trên đúng `data/conversations.json` và `data/advanced_long_context.json` của repo.

## Standard Benchmark (10 hội thoại ngắn, ~10 turn/hội thoại)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------:|--------------------------:|----------------------:|-------------------:|------------------------:|-------------:|
| Baseline |               6763 |                     67400 |                   0.00 |                0.60 |                       0 |            0 |
| Advanced |               9486 |                     80486 |                   1.00 |                1.00 |                     208 |           51 |

## Long-Context Stress Benchmark (1 hội thoại rất dài, nhiều correction + nhiễu)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------:|--------------------------:|----------------------:|-------------------:|------------------------:|-------------:|
| Baseline |               1172 |                     90951 |                   0.00 |                0.60 |                       0 |            0 |
| Advanced |               5615 |                     47517 |                   1.00 |                1.00 |                     161 |           26 |

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ state trong `SessionState` theo `thread_id`; nó không có khái niệm "user" độc lập với "thread". Khi câu hỏi recall được hỏi ở một `thread_id` mới (đúng định nghĩa "cross-session"), session của Baseline trống — nó chỉ có thể trả lời nếu may mắn tìm được overlap từ trước trong **cùng** thread, và trong test này luôn là 0.00 ở cả hai benchmark.

Advanced tách fact ổn định (tên, nơi ở, nghề nghiệp, sở thích, style...) ra khỏi thread và ghi vào `User.md` qua `UserProfileStore`, độc lập với `thread_id`. Khi sang thread mới, `_offline_response` đọc lại `profile_store.facts(user_id)` — không phụ thuộc lịch sử của thread cũ — nên recall đạt 1.00 tuyệt đối ở cả hai bộ dữ liệu, kể cả khi dữ liệu cố tình chèn nhiễu (Hà Nội, "product manager" chỉ là chuyện đùa) và correction (Huế → Đà Nẵng, backend → MLOps).

## 2. Vì sao Advanced tốn hơn ở hội thoại ngắn?

Ở Standard Benchmark, Advanced tốn **nhiều hơn** Baseline ở cả hai cột token (9486 vs 6763 token agent; 80486 vs 67400 token prompt). Nguyên nhân:

- Mỗi turn, Advanced phải chạy regex extraction (`extract_profile_updates_with_confidence`), rồi cộng `estimate_tokens(profile_text) + estimate_tokens(summary) + tokens(kept messages)` làm phần "prompt context" — tức là nó luôn cộng thêm chi phí đọc `User.md` mà Baseline không có.
- Với hội thoại ngắn (10 turn/conversation), `CompactMemoryManager` chưa kịp "trả lại" lợi ích nén vì threshold (800 token mặc định) thường chưa vượt nhiều, trong khi chi phí cố định (đọc profile, ghi fact) đã phải trả ngay từ turn đầu.
- Đây chính là điểm Guide.md nhấn mạnh: **memory có cấu trúc là một khoản đầu tư trả trước (upfront cost)** — nó không miễn phí, và ở quy mô nhỏ, khoản đầu tư này chưa thu hồi được.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Ở Stress Benchmark, thứ tự đảo ngược hoàn toàn ở cột quan trọng nhất — **Prompt tokens processed**: Advanced (47517) chỉ bằng **~52%** của Baseline (90951).

- Baseline không có compaction (`compactions = 0` luôn, theo định nghĩa). Mỗi turn nó phải re-estimate token trên **toàn bộ lịch sử thô** của thread (`session.prompt_tokens_processed += sum(estimate_tokens(m) for m in toàn bộ session.messages)`), nên chi phí này tăng gần như bậc hai theo số turn (turn thứ N phải "trả lại" cả N-1 turn trước).
- Advanced compact 26 lần trong đúng 1 thread dài này: mỗi khi vượt `threshold_tokens`, nó nén phần cũ thành 1 đoạn summary ngắn và chỉ giữ `keep_messages` (mặc định 6) message gần nhất ở dạng đầy đủ. Vì vậy `_estimate_prompt_context_tokens` luôn bị **chặn trên** bởi `profile + summary + keep_messages` — không tăng vô hạn theo độ dài thread.
- `Agent tokens only` của Advanced (5615) vẫn cao hơn Baseline (1172) vì câu trả lời của Advanced dài hơn (liệt kê đủ fact), nhưng đây là chi phí nhỏ so với khoản tiết kiệm khổng lồ ở `Prompt tokens processed` — và quan trọng hơn, chỉ Advanced mới recall đúng (1.00 vs 0.00).

Đây đúng là câu chuyện cốt lõi track muốn minh họa: **compact tối ưu chủ yếu ở "ngữ cảnh phải mang theo qua các lượt" (prompt tokens), không phải ở số token Output agent tự sinh ra.**

## 4. Memory file tăng trưởng ra sao, rủi ro gì?

`User.md` tăng 208 byte (Standard) và 161 byte (Stress) cho một user duy nhất qua 10 (hoặc 1 hội thoại rất dài) lượt hội thoại — tăng trưởng tuyến tính theo số fact **khác nhau**, không theo số turn, vì `upsert_fact` ghi đè tại chỗ (idempotent) thay vì append vô hạn.

Rủi ro đi kèm:

- **Phình theo số user, không theo số turn của 1 user**: với hệ thống thật có hàng nghìn user, tổng dung lượng `state/profiles/*.md` tăng tuyến tính theo số user × số fact/user — cần giới hạn số fact tối đa hoặc archive định kỳ.
- **Ghi sai fact = lỗi "vĩnh viễn"**: vì `upsert_fact` ghi đè theo `key`, một fact bị extract sai (ví dụ bug đã gặp: `"...mình nuôi con gì."` từng bị parse thành `pet=gì`) sẽ **xoá mất** giá trị đúng trước đó cho đến khi có correction mới — khác với lỗi trong 1 câu trả lời đơn lẻ (chỉ ảnh hưởng 1 turn), lỗi ghi nhớ ảnh hưởng **mọi turn sau đó** cho tới khi sửa.
- **Không có cơ chế hết hạn (decay)**: một sở thích/preference cũ nói 6 tháng trước vẫn được coi ngang hàng với fact nói hôm nay, trừ khi có correction rõ ràng đè lên — đây là hướng mở rộng "memory decay" được liệt trong Rubric nhưng lab này chưa làm.

## 5. Bonus đã làm: Confidence threshold (`src/memory_store.py`, `FACT_CONFIDENCE`)

**Vấn đề giải quyết:** không phải pattern regex nào cũng đáng tin cậy như nhau. Trong quá trình build, chính `_PET_RE` (`"nuôi (?:một |)(?:bé |con |)([a-zà-ỹ]+)"`) đã từng match nhầm câu hỏi `"...mình nuôi con gì."` thành fact `pet="gì"`, ghi đè giá trị đúng `"corgi"`. Lỗi đó được sửa tận gốc bằng `RECALL_REQUEST_RE` (bỏ qua toàn bộ câu hỏi/mệnh lệnh recall trước khi extract), nhưng về nguyên tắc, các pattern lỏng hơn (`pet`, `interest`) vẫn có nguy cơ false-positive cao hơn các pattern có anchor rõ ràng (`"đồ uống yêu thích là X"`, `"món ăn yêu thích là X"`).

**Cách hoạt động:** mỗi fact key có một confidence cố định (0.6–0.95, xem `FACT_CONFIDENCE`). `AdvancedAgent` gọi `extract_profile_updates_with_confidence(message, min_confidence=PROFILE_CONFIDENCE_THRESHOLD)` (mặc định `0.5`) trước khi `upsert_fact` — fact có confidence thấp hơn ngưỡng sẽ bị bỏ qua, không bao giờ chạm tới `User.md`.

**Cải thiện gì / đánh đổi gì:**
- Ở mức ngưỡng mặc định `0.5`, **không có thay đổi** so với benchmark gốc (toàn bộ fact hiện tại đều ≥ 0.6) — bonus này là một **lưới an toàn cấu hình được**, không phải một thay đổi hành vi mặc định.
- Nếu tăng ngưỡng lên ví dụ `0.65`, fact `pet` (0.6) sẽ bị loại hoàn toàn khỏi `User.md` — đổi lại, hệ thống **an toàn hơn** trước các pattern lỏng nhưng **mất khả năng nhớ** những fact loại đó cho đến khi pattern được làm chắc hơn (ví dụ: yêu cầu fact xuất hiện ≥ 2 lần độc lập mới tính là "đủ tin cậy" — hướng mở rộng tiếp theo).
- Rủi ro: ngưỡng đặt sai (quá cao) sẽ làm giảm recall một cách "im lặng" — agent sẽ không báo lỗi, chỉ đơn giản là không nhớ fact đó. Cần benchmark lại mỗi khi đổi ngưỡng để tránh regression không nhận ra.
