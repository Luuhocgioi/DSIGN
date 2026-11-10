# Ôn tập vấn đáp Deep Learning cho dự án nhận diện ngôn ngữ ký hiệu Việt Nam

Tài liệu bám theo mã nguồn và lần huấn luyện mô hình **“chào”** và **“tôi yêu bạn”** trong dự án Sign Language AI. Hai cụm này là hai nhãn độc lập. “Tôi yêu bạn” được lưu dưới dạng một token nguyên cụm. Các con số dưới đây thuộc bản Transformer hai nhãn trên CPU.

## 1. Tổng quan dự án

**Bài toán nhận diện.** Hệ thống nhận chuỗi khung hình từ webcam, trích xuất tọa độ bàn tay và tư thế, rồi nhận diện ký hiệu từ chuyển động theo thời gian. Kết quả được hiển thị dưới dạng văn bản tiếng Việt, có thể ghi transcript hoặc đọc bằng giọng hệ thống. Hệ thống có thể ghép các ký hiệu đã chốt theo thời gian. Bản thực nghiệm hiện tại chỉ kiểm chứng hai nhãn đóng, chưa xác lập khả năng dịch ngôn ngữ ký hiệu liên tục.

**Bài toán tra cứu.** Người dùng nhập tiếng Việt vào Streamlit và xem clip khớp nhãn. Tính năng này đối chiếu `label.csv` với tệp MP4, ưu tiên khớp trực tiếp rồi tìm không dấu hoặc gợi ý gần đúng. Kho tra cứu có 4.362 video và 3.314 nhãn sau chuẩn hóa hoa thường. Luồng này độc lập với quá trình suy luận của Transformer.

```text
Huấn luyện: metadata CSV → NPZ → tăng cường/batch → Transformer → loss → checkpoint
Webcam:     OpenCV → MediaPipe → cửa sổ landmark → Transformer → CTC → ổn định → văn bản
Tra cứu:    từ hoặc cụm từ → nhãn trong CSV → chọn video → phát MP4
```

**Câu giới thiệu khoảng 40 giây.** “Đề tài của em nhận diện ký hiệu Việt Nam qua webcam. Em dùng MediaPipe lấy 201 tọa độ trên mỗi khung, sau đó đưa chuỗi khung vào Transformer Encoder để học diễn biến của động tác. Mô hình xuất xác suất theo khung và dùng CTC để học từ nhãn của cả clip mà không cần đánh dấu vị trí ký hiệu ở từng khung. Bản hiện tại được huấn luyện từ đầu với hai nhãn ‘chào’ và ‘tôi yêu bạn’. Kết quả là 18/18 mẫu đúng trên tập test landmark. Em chưa đo được độ chính xác với người ký mới qua webcam.”

## 2. Cơ sở lý thuyết cần nắm

1. **Neuron và lớp tuyến tính.** Một neuron tính tổng có trọng số cộng bias, sau đó có thể đi qua hàm kích hoạt. Lớp Linear tính `y = Wx + b` cho nhiều neuron cùng lúc. Một chuỗi chỉ gồm các phép affine vẫn gộp được thành một phép affine, nên mạng cần tính phi tuyến để biểu diễn quan hệ phức tạp.
2. **Activation.** ReLU đặt giá trị âm về 0. GELU điều chỉnh đầu vào trơn hơn, với dạng `GELU(x) = x Φ(x)`, trong đó Φ là hàm phân phối tích lũy Gaussian chuẩn. Dự án dùng GELU giữa hai lớp Linear trong feed-forward. Softmax dùng để chuẩn hóa điểm thành xác suất. LogSoftmax trả về log-xác suất ổn định hơn về số học, phù hợp đầu vào của CTC.
3. **Loss và metric.** Loss là mục tiêu tối ưu có gradient. Accuracy, CER và WER được đo sau giải mã và dùng để diễn giải chất lượng. Dự đoán vẫn có thể đúng khi loss tăng nếu phân bố xác suất dành ít khối lượng hơn cho các đường căn chỉnh đúng.
4. **Forward pass.** Đưa input qua từng lớp để tạo log-xác suất, dùng nhãn và độ dài chuỗi để tính loss. Ở bước này, trọng số dùng để tính toán và chưa được cập nhật.
5. **Backpropagation.** Dùng quy tắc dây chuyền để tính đạo hàm loss theo từng tham số. PyTorch giữ đồ thị tính toán trong huấn luyện và thực hiện quá trình đó khi gọi `backward()`. Sau khi có gradient, optimizer mới cập nhật trọng số.
6. **Gradient descent và AdamW.** SGD cơ bản cập nhật theo hướng âm của gradient với learning rate. AdamW dùng trung bình động của gradient và bình phương gradient để điều chỉnh bước cập nhật theo tham số, đồng thời tách weight decay khỏi phép cập nhật thích nghi. Learning rate quá lớn có thể làm loss dao động, còn quá nhỏ có thể khiến mô hình học chậm.
7. **CNN và temporal CNN.** CNN ảnh dùng bộ lọc cục bộ để nhận mẫu không gian. Nếu phân loại từng khung độc lập, nó không mô tả trực tiếp thứ tự động tác. CNN 1D theo thời gian có thể xử lý chuỗi landmark hiệu quả. Phạm vi quan sát của nó phụ thuộc kernel, độ sâu và dilation.
8. **RNN, LSTM và GRU.** RNN truyền trạng thái ẩn theo bước thời gian. LSTM và GRU dùng cổng để kiểm soát thông tin giữ lại, hỗ trợ học quan hệ dài hơn RNN cơ bản. Đây là baseline hợp lý với dữ liệu nhỏ. Việc tính trạng thái theo thứ tự làm hạn chế song song hóa theo thời gian.
9. **Transformer.** Self-attention cho mỗi khung kết hợp thông tin từ những khung khác trong cửa sổ. Các biểu diễn thời gian có thể tính song song trong từng tầng. Chi phí của attention đầy đủ tăng theo bình phương chiều dài chuỗi, nên cửa sổ cố định hữu ích khi chạy webcam.
10. **Baseline Machine Learning.** SVM hoặc Random Forest có thể dùng đặc trưng như góc khớp, vận tốc, độ lệch chuẩn hoặc vector chuỗi đã cố định độ dài. DTW có thể so sánh quỹ đạo với mẫu tham chiếu khi tốc độ ký khác nhau. Với hai lớp và ít video độc lập, các baseline này đáng thử. Dự án chọn Transformer để học quan hệ thời gian từ landmark, chưa có phép so sánh đối chứng để kết luận Transformer vượt các baseline.

## 3. Kiến trúc Transformer và số tham số

### 3.1. Kích thước tensor và vai trò từng lớp

| Bước | Kích thước | Vai trò |
|---|---|---|
| Đầu vào | `[B, T, 201]` | `B` là số mẫu trong batch, `T` là số khung đã đệm |
| Linear 201 → 64 và LayerNorm | `[B, T, 64]` | Chiếu tọa độ sang không gian đặc trưng và chuẩn hóa theo 64 kênh tại mỗi vị trí |
| Positional encoding + dropout | `[B, T, 64]` | Cộng vector sin/cos theo vị trí để biểu diễn thứ tự khung |
| Hai tầng Transformer Encoder | `[B, T, 64]` | Trộn thông tin theo thời gian bằng attention rồi biến đổi từng vị trí bằng feed-forward |
| Linear 64 → 4 | `[B, T, 4]` | Sinh logits cho blank, unknown và hai nhãn |
| LogSoftmax và hoán vị trục | `[T, B, 4]` | Tạo log-xác suất theo lớp và đưa về thứ tự trục CTC yêu cầu |

**Attention.** Từ biểu diễn đầu vào, mô hình tạo Query (Q), Key (K) và Value (V). Điểm liên hệ giữa một query và các key được tính từ tích vô hướng, chia cho căn bậc hai chiều mỗi head, sau đó lấy softmax. Trọng số này được dùng để lấy tổng có trọng số của V. Bản hai nhãn có 4 head và `d_model = 64`, nên mỗi head có 16 chiều, hệ số chia là √16 = 4. Padding mask chặn các vị trí đệm. Không có causal mask, nên một vị trí có thể sử dụng các khung ở cả hai phía **bên trong cửa sổ hiện có**.

**Một tầng Encoder.** Mô hình dùng pre-norm (`norm_first=True`). Mỗi tầng có hai nhánh với kết nối tắt. Nhánh thứ nhất chuẩn hóa rồi self-attention, nhánh thứ hai chuẩn hóa rồi dùng feed-forward `64 → 128 → GELU → 64`. Các kết nối tắt cộng đầu ra nhánh vào đầu vào, giúp giữ thông tin và hỗ trợ truyền gradient.

**LayerNorm, BatchNorm và dropout.** LayerNorm chuẩn hóa theo chiều đặc trưng của từng token nên không cần ước lượng thống kê từ các mẫu khác trong batch. BatchNorm thường dựa trên thống kê theo batch và cần lưu thống kê dùng khi suy luận. Dự án dùng LayerNorm, không có lớp BatchNorm. Dropout 0,1 che ngẫu nhiên một phần giá trị khi huấn luyện và tự tắt khi gọi `eval()`. Weight decay của AdamW là 0,0001. Đây là cơ chế regularization riêng với dropout.

### 3.2. Tính đúng 80.260 tham số học được

| Thành phần | Phép tính | Tham số |
|---|---:|---:|
| Linear đầu vào | 201 × 64 + 64 | 12.928 |
| LayerNorm đầu vào | 64 hệ số scale + 64 bias | 128 |
| Q, K, V và chiếu đầu ra của attention, mỗi tầng | 4 × (64 × 64 + 64) | 16.640 |
| Hai lớp Linear của feed-forward, mỗi tầng | (64 × 128 + 128) + (128 × 64 + 64) | 16.576 |
| Hai LayerNorm, mỗi tầng | 2 × (64 + 64) | 256 |
| Hai tầng Encoder | 2 × (16.640 + 16.576 + 256) | 66.944 |
| Linear phân loại | 64 × 4 + 4 | 260 |
| **Tổng cộng** | **12.928 + 128 + 66.944 + 260** | **80.260** |

GELU, LogSoftmax, dropout, kết nối tắt và mã hóa vị trí sin/cos không thêm tham số học được. Bảng tính theo checkpoint hai nhãn. Tăng số head trong khi giữ chiều mô hình và cách chiếu Q/K/V như hiện tại sẽ đổi cách chia chiều theo head, nhưng không tự làm tăng số tham số của attention.

### 3.3. CTC, blank và loss phụ

1. **Tại sao dùng CTC?** Mỗi mẫu chỉ có nhãn cả clip. CTC cộng xác suất của các đường đi theo khung có thể quy về nhãn đích, thay vì cần biết chính xác khung nào thuộc ký hiệu.
2. **Quy tắc quy gọn.** Gom các token giống nhau liền kề rồi bỏ blank. Ví dụ `blank, A, A, blank` cho `A`. Chuỗi `A, blank, A` cho `A, A`. Blank có chức năng căn chỉnh, khác với nhãn unknown và khác với một lớp nền được thu thập để đánh giá webcam.
3. **Giải mã greedy.** Ý tưởng cơ bản là chọn token có điểm cao nhất ở mỗi khung rồi quy gọn. Trong mã dự án, decoder còn có quy tắc chọn nhãn trội cho chuỗi đủ dài và dự phòng nhãn non-blank khi đường greedy rỗng. Luồng webcam còn dùng các điều kiện lọc riêng. Kết quả của đường greedy không nhất thiết là chuỗi có xác suất tổng lớn nhất.
4. **Loss đúng theo mã nguồn.** Với batch của bản hai nhãn, `L_train = 0,65 × L_CTC + 0,35 × L_CE_phụ`. Trainer chỉ tính CE phụ khi mọi mẫu trong batch có đúng một token đích. CE phụ gộp log-xác suất trên các khung thật bằng logsumexp trừ log số khung, rồi dùng cross-entropy với label smoothing 0,05. Nhật ký `train_loss` và `val_loss` ghi **CTC loss**, còn tối ưu huấn luyện dùng tổ hợp có CE phụ. Khi trình bày, dùng công thức theo mã nguồn thay cho cách viết tắt “CTC + 0,35 × CE” trong báo cáo.
5. **Phương án đơn giản hơn.** Với bài toán clip một nhãn, lấy trung bình biểu diễn theo thời gian rồi dùng một đầu phân loại CE là baseline hợp lý. CTC thích hợp khi cần căn chỉnh mà không có nhãn từng khung hoặc muốn mở rộng lên chuỗi token. Bản hiện tại chưa chứng minh chất lượng trên chuỗi nhiều ký hiệu.

## 4. Dữ liệu và quy trình huấn luyện

### 4.1. Tập dữ liệu và xử lý đặc trưng

| Nhãn | Train | Validation | Test |
|---|---:|---:|---:|
| chào | 32 | 4 | 5 |
| tôi yêu bạn | 98 | 11 | 13 |
| **Tổng** | **130** | **15** | **18** |

1. **Lập metadata.** Mỗi dòng liên kết tệp `feature_path` và nhãn `text`. Lần huấn luyện này đọc đặc trưng NPZ sẵn có, mỗi mảng đã kiểm tra có dạng `[60, 201]` và chứa giá trị hữu hạn. Đã loại một mảng validation trùng tuyệt đối khi so mã băm với các mảng đã duyệt.
2. **Đặc trưng webcam.** Layout `vsl_201` ghép 25 điểm tư thế, 21 điểm tay trái và 21 điểm tay phải. Mỗi điểm có x, y, z, nên `(25 + 21 + 21) × 3 = 201`. Layout này dùng trực tiếp tọa độ MediaPipe, không thêm điểm mặt hay cờ hiện diện riêng. Nhóm điểm bị thiếu được điền 0. Một nhóm thiếu và một khung padding là hai khái niệm khác nhau. Padding mask lấy từ độ dài chuỗi.
3. **Mã hóa nhãn.** ID 0 là blank, ID 1 là unknown, ID 2 và 3 là hai nhãn thật theo thứ tự từ vựng. Tra cụm nguyên văn trước khi tách từ nên “tôi yêu bạn” có đúng một ID. WER vẫn tính trên các từ của văn bản đã giải mã.
4. **Chia tập.** Train cập nhật trọng số. Validation chọn checkpoint và hỗ trợ điều chỉnh thiết lập. Test đo sau khi chốt mô hình. Loại mảng giống hệt không bảo đảm tách theo người ký hoặc clip nguồn. Kho MP4 có 1 clip đúng nhãn “chào” và 3 clip đúng nhãn “tôi yêu bạn”. Chưa có ánh xạ từ từng NPZ về clip gốc để xác minh tính độc lập.
5. **Tăng cường.** Cấu hình dùng nhiễu Gaussian, scale, tịnh tiến, xoay, thay đổi thời gian, bỏ khung và che điểm. Ví dụ Gaussian có độ lệch chuẩn 0,008, khoảng scale 0,92–1,08 và xác suất che điểm 0,02. Chỉ bật cho train. Biến đổi quá mạnh có thể làm đổi đặc trưng phân biệt của ký hiệu. Cần kiểm tra bằng mẫu thực tế.
6. **Cân bằng và padding.** Lấy mẫu có trọng số hỗ trợ nhãn ít mẫu hơn trong train. Nó không tạo thêm người ký độc lập. Khi ghép batch, các chuỗi có độ dài khác nhau được đệm đến chiều dài lớn nhất, CTC nhận độ dài thật của từng input và từng target.

### 4.2. Một bước huấn luyện

1. Chọn batch, đưa tensor vào thiết bị và xóa gradient tích lũy từ bước trước.
2. Chạy forward qua Transformer, thu `[T, B, 4]`.
3. Tính CTC từ log-xác suất, nhãn và các độ dài. Với batch một token mỗi mẫu, tính thêm CE phụ và gộp theo tỉ lệ 65/35.
4. Gọi `backward()` để tính gradient. Trainer giới hạn chuẩn gradient tổng ở 1,0 trước bước cập nhật để giảm tác động của gradient lớn.
5. AdamW cập nhật trọng số. Lần chạy dùng CPU nên AMP dành cho CUDA không được bật.
6. Sau mỗi epoch, chạy validation không cập nhật trọng số. Scheduler ReduceLROnPlateau theo dõi validation loss. Chọn best checkpoint theo accuracy validation rồi dùng validation loss để phân xử khi bằng accuracy.

**Cấu hình đã dùng.** Seed 42, CPU, batch 16, learning rate ban đầu 0,0003, weight decay 0,0001, dropout 0,1, hai tầng Encoder. Tệp YAML có mặc định 30 epoch, nhưng lần chạy dùng `--epochs 20`. Các giá trị này là cấu hình đã chạy. Chưa có log ablation hoặc tìm kiếm siêu tham số có hệ thống để chứng minh chúng tối ưu.

### 4.3. Suy luận qua webcam

1. `run_camera.bat` chọn checkpoint hai nhãn và camera số 0. Đã xác minh camera đọc được hình 640 × 480 qua DirectShow.
2. Luồng chính thu và hiển thị hình, worker nền xử lý landmark. Hàng đợi giữ tối đa một khung chờ và ưu tiên khung mới khi bị dồn, giúp giảm độ trễ do xếp hàng.
3. Mô hình nhận cửa sổ 60 khung đã xử lý. Cấu hình đặt bước dự đoán 2 và yêu cầu đủ khung có bàn tay. 30 FPS là mục tiêu cấu hình camera. Vì worker có thể bỏ khung, không thể suy ra độ trễ thực tế chỉ từ `60 / 30`.
4. Ứng dụng lọc dự đoán ở ngưỡng 0,45 và cần 3 phiếu trong 5 lần dự đoán để có ứng viên ổn định. Ký hiệu có thể được chốt tự động theo trạng thái thời gian/chuyển động hoặc bằng Enter. Ngưỡng là tham số vận hành, không phải độ tin cậy đã được hiệu chuẩn.
5. Hậu xử lý chuẩn hóa hiển thị văn bản và có chỗ cấu hình luật sắp xếp cụm. Bản hai nhãn không có luật sắp xếp được cấu hình. pyttsx3 phát âm qua giọng hệ thống, tùy giọng được cài trên Windows.

## 5. Kết quả và cách diễn giải

| Epoch | Train CTC loss | Validation CTC loss | Validation accuracy |
|---:|---:|---:|---:|
| 1 | 14,3925 | 1,9951 | 26,67% |
| 4 | 1,7384 | 1,6390 | 100,00% |
| 19 | 0,8437 | 0,3325 | 100,00% |
| 20 | 0,6864 | 0,7207 | 100,00% |

1. **Chọn checkpoint.** Epoch 19 và 20 cùng đúng 15/15 mẫu validation. Epoch 19 có validation loss thấp hơn, nên dùng để đánh giá test. Lưu riêng best và last để phân biệt mô hình triển khai với trạng thái cuối của quá trình huấn luyện.
2. **Kết quả test.** Checkpoint epoch 19 đạt 18/18 mẫu khớp nhãn, gồm 5/5 “chào” và 13/13 “tôi yêu bạn”. CER và WER bằng 0. Test CTC loss xấp xỉ 0,34945. Chưa đo độ chính xác webcam, độ trễ hay hiệu năng trên người ký mới.
3. **Accuracy.** Mã so sánh cả nhãn sau chuẩn hóa Unicode NFC, bỏ khoảng trắng ở hai đầu và đổi chữ hoa. Accuracy bằng số mẫu khớp toàn bộ chia tổng số mẫu. Nếu luôn đoán lớp đông nhất trên tập test này, baseline đạt 13/18 ≈ 72,22%, dù không nhận diện đúng mẫu “chào” nào.
4. **Precision, recall, F1.** Xét một nhãn là dương tính: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). Với hai nhãn, có thể tính riêng từng lớp rồi lấy macro-F1 để hai lớp có trọng số ngang nhau. Báo cáo JSON thực nghiệm hiện không xuất các chỉ số này. Khi thảo luận, phân biệt công thức và chỉ số đã ghi nhận trong tệp.
5. **CER và WER.** Cộng số phép chèn, xóa và thay nhỏ nhất rồi chia cho tổng số ký tự hoặc từ tham chiếu. Ví dụ tham chiếu “tôi yêu bạn”, dự đoán “tôi bạn” có một từ bị xóa, WER = 1/3. Vì mã so sánh sau giải mã, WER là cấp từ văn bản, không phải cấp token của mô hình.
6. **Loss curve.** Train CTC loss giảm từ 0,8437 xuống 0,6864 ở epoch 20, trong khi validation tăng từ 0,3325 lên 0,7207. Đây là tín hiệu mô hình có thể tổng quát kém hơn ở epoch cuối, chưa đủ để kết luận overfitting kéo dài. Accuracy không đổi vì nhãn argmax vẫn đúng dù xác suất dành cho đường đi CTC hợp lệ giảm.
7. **Underfitting và overfitting.** Underfitting thường có lỗi cao cả ở train lẫn validation. Kiểm tra bằng việc xác minh pipeline, khả năng học một tập rất nhỏ, thời gian huấn luyện và dung lượng mô hình. Overfitting thường xuất hiện khi train tiếp tục tốt lên nhưng validation xấu đi kéo dài. Có thể thu thêm dữ liệu độc lập, tăng regularization hợp lý, giảm dung lượng hoặc chọn checkpoint sớm hơn.

## 6. Lý do chọn kỹ thuật và hướng phát triển

| Quyết định | Lý do phù hợp | Cách đối chứng |
|---|---|---|
| Landmark thay cho RGB thô | Giảm số chiều, tập trung vào tư thế và chuyển động. Phụ thuộc vào chất lượng phát hiện điểm | So sánh cùng split với đầu vào RGB hoặc thêm đặc trưng góc khớp |
| Transformer Encoder nhỏ | Học liên hệ giữa các khung trong cửa sổ. 80.260 tham số là quy mô của bản thực nghiệm | Đo accuracy, macro-F1 và độ trễ với DTW, GRU hoặc temporal CNN |
| CTC và CE phụ | Dùng nhãn cấp clip, giữ khả năng học căn chỉnh. CE phụ cung cấp tín hiệu phân loại cả chuỗi | So CTC thuần, tổ hợp hiện tại và pooling + CE dưới cùng điều kiện |
| LayerNorm và AdamW | LayerNorm không phụ thuộc thống kê batch. AdamW cho bước thích nghi và weight decay tách rời | Thử đổi từng yếu tố trên validation, giữ split và ngân sách huấn luyện |
| OpenCV, MediaPipe, PyTorch, Streamlit | Lần lượt phụ trách camera, landmark, mô hình và tra cứu video | Đo tách từng bước để tìm nút thắt thay vì quy toàn bộ độ trễ cho Transformer |

1. **Độc lập dữ liệu.** Thu thêm video từ người ký khác, lưu mã người và clip nguồn, tách train/validation/test theo nhóm trước khi tăng cường. Cỡ tập test hiện nhỏ, mỗi lỗi làm accuracy giảm khoảng 5,56 điểm phần trăm.
2. **Chống chốt nhầm.** Đánh giá các tình huống không ký, động tác ngoài từ vựng, thiếu tay và chuyển động nhanh. Có thể học cơ chế từ chối hoặc lớp nền từ dữ liệu phù hợp, rồi chỉnh ngưỡng trên validation. Blank CTC tự nó không đo được khả năng từ chối ký hiệu lạ.
3. **Nhiều nhãn và câu liên tục.** Mở rộng nhãn, cập nhật từ vựng và đầu ra, sau đó huấn luyện lại. Muốn nhận diện câu liên tục, cần clip nhiều ký hiệu với bản chép theo thứ tự và đánh giá phân đoạn, lặp, bỏ sót và trật tự token. Việc thêm nhãn vào CSV tra cứu chỉ mở rộng kho video.
4. **Đo thực tế.** Ghi số lượt đúng, sai, bỏ sót, chốt nhầm và độ trễ từ lúc kết thúc ký tới lúc có kết quả ổn định. Báo cáo theo người ký và điều kiện quay. Đo FPS thu hình, FPS xử lý và phân vị độ trễ riêng.
5. **Tuning có kiểm soát.** Thử learning rate, độ dài cửa sổ, dropout và kích thước mô hình trên validation. Giữ test cho đánh giá cuối cùng. Ghi cấu hình, checkpoint, seed, split và kết quả của mỗi lần chạy.

## 7. Bộ 42 câu hỏi và đáp án mẫu

Mỗi câu nên trả lời theo ba nhịp: nêu khái niệm, liên hệ với dự án, chốt bằng số liệu hoặc giới hạn đã xác minh.

1. **Đề tài giải quyết bài toán gì?** Nhận chuỗi hình webcam để xác định một trong hai ký hiệu “chào” hoặc “tôi yêu bạn” và hiển thị văn bản. Luồng Streamlit giải quyết bài toán khác là tra cứu video theo nhãn.
2. **Input và output của mô hình là gì?** Input là `[B,T,201]` landmark. Output là `[T,B,4]` log-xác suất, sau đó giải mã ra token.
3. **Vì sao một ảnh tĩnh có thể không đủ?** Hai động tác có thể chia sẻ tư thế trung gian nhưng khác thứ tự hoặc hướng chuyển động. Chuỗi khung giữ được diễn biến.
4. **Vì sao dùng Deep Learning thay vì SVM?** Transformer học biểu diễn và quan hệ theo thời gian từ chuỗi điểm. SVM vẫn là baseline nếu có đặc trưng tổng hợp phù hợp. Em chưa đo đối chứng nên chưa thể khẳng định Transformer tốt hơn.
5. **MediaPipe và Transformer học gì?** MediaPipe cung cấp landmark đã trích xuất. Transformer được huấn luyện để ánh xạ chuỗi landmark sang nhãn của dự án. Lần chạy này không huấn luyện lại MediaPipe.
6. **201 chiều từ đâu?** 25 điểm tư thế, 21 điểm tay trái, 21 điểm tay phải, mỗi điểm có x, y, z. Tổng cộng 67 × 3.
7. **MediaPipe không thấy tay thì sao?** Nhóm tay được điền 0. Webcam còn kiểm tra có đủ khung thấy tay trước khi dự đoán. Mất điểm có thể làm nhận diện sai hoặc bỏ sót.
8. **Vì sao “tôi yêu bạn” có một token?** Từ vựng lưu nguyên cụm và tokenizer khớp nguyên cụm trước. Nhãn mô hình đại diện cho toàn bộ ký hiệu/cụm trong bộ dữ liệu.
9. **Kích thước train, validation, test là bao nhiêu?** 130, 15 và 18. Từng nhãn lần lượt là 32/4/5 và 98/11/13.
10. **Mục đích ba tập khác nhau thế nào?** Train cập nhật trọng số, validation chọn checkpoint hoặc tinh chỉnh, test dùng để đánh giá sau khi chốt mô hình.
11. **Em có bảo đảm không rò rỉ dữ liệu không?** Em đã loại mảng trùng tuyệt đối theo mã băm. Chưa có mã người ký và ánh xạ clip nguồn nên chưa thể bảo đảm tách độc lập.
12. **Data augmentation nào được dùng?** Nhiễu tọa độ, scale, tịnh tiến, xoay, biến đổi thời gian, bỏ khung và che điểm. Chỉ áp dụng train để test vẫn phản ánh dữ liệu có sẵn.
13. **Lấy mẫu cân bằng có thay thế việc thu dữ liệu không?** Không. Nó tăng cơ hội thấy lớp ít mẫu trong batch, nhưng không thêm người ký, góc quay hay biến thể thật.
14. **Padding và padding mask khác gì?** Padding thêm khung giả để tạo tensor cùng kích thước. Mask báo cho attention bỏ qua vị trí đệm. CTC dùng độ dài thật để giới hạn chuỗi.
15. **Linear đầu tiên làm gì và có bao nhiêu tham số?** Chiếu mỗi khung từ 201 chiều sang 64 chiều, có 201 × 64 trọng số và 64 bias, tổng 12.928.
16. **Nếu bỏ mọi activation khỏi FFN thì sao?** Hai phép Linear nối tiếp gộp được thành một phép affine. Attention vẫn phi tuyến, nhưng nhánh FFN giảm khả năng biểu diễn.
17. **Mô hình dùng GELU ở đâu?** Giữa Linear 64 → 128 và Linear 128 → 64 trong mỗi tầng Encoder.
18. **Positional encoding có tác dụng gì?** Cung cấp thứ tự cho các khung để attention phân biệt diễn biến khác nhau. Dạng sin/cos được tính sẵn, không học thêm tham số.
19. **Giải thích Q, K và V.** Q biểu thị thông tin vị trí đang tìm, K dùng để tính độ liên quan, V là nội dung được tổng hợp theo trọng số attention.
20. **Tại sao chia điểm attention cho √dₖ?** Tích vô hướng có thể lớn khi chiều tăng, khiến softmax quá sắc. Chia giúp kiểm soát độ lớn của logits. Với 16 chiều, hệ số là 4.
21. **Bốn head có làm tham số attention tăng gấp bốn không?** Với d_model giữ nguyên, không. Mỗi head làm việc trên một phần chiều, tổng chiều chiếu vẫn là 64.
22. **LayerNorm khác BatchNorm ra sao?** LayerNorm dùng thống kê theo các kênh của từng vị trí. BatchNorm phụ thuộc vào thống kê batch khi train và thường dùng running statistics khi eval.
23. **Tác dụng của residual và pre-norm?** Residual tạo đường tắt cho thông tin và gradient. Pre-norm chuẩn hóa trước nhánh attention hoặc FFN, rồi cộng đầu ra nhánh vào đầu vào.
24. **Mô hình có tổng bao nhiêu tham số?** 80.260, gồm 13.056 ở chiếu đầu vào và LayerNorm, 66.944 ở hai tầng Encoder và 260 ở đầu phân loại.
25. **Vì sao có 4 đầu ra khi chỉ nhận 2 ký hiệu?** Hai ID dành cho blank và unknown. Hai ID còn lại dành cho nhãn thật.
26. **Vì sao chọn CTC?** Có nhãn cả clip nhưng không có mốc từng khung. CTC cộng xác suất trên các đường căn chỉnh hợp lệ.
27. **Blank khác unknown thế nào?** Blank hỗ trợ căn chỉnh thời gian và bị bỏ khi giải mã. Unknown là token từ vựng cho nhãn chưa biết, không tự tạo khả năng từ chối động tác lạ.
28. **CTC xử lý hai token giống nhau liên tiếp thế nào?** Trong đường đi theo khung cần một blank ngăn cách để quy về hai token giống nhau. `A, A` cho một `A`, còn `A, blank, A` cho hai `A`.
29. **Loss dùng huấn luyện chính xác là gì?** 0,65 CTC + 0,35 CE phụ với batch một token đích mỗi mẫu. Training CSV ghi CTC loss, không ghi giá trị loss hỗn hợp.
30. **Forward, backward và optimizer step khác nhau thế nào?** Forward tính đầu ra và loss. Backward tính gradient. Optimizer step thay đổi tham số theo gradient và trạng thái optimizer.
31. **AdamW khác SGD như thế nào?** AdamW dùng thống kê động của gradient và bình phương gradient, cùng weight decay tách rời. SGD cơ bản dùng gradient hiện tại và một learning rate chung.
32. **Gradient clipping khác weight decay ra sao?** Clipping giới hạn chuẩn gradient khi cập nhật để tránh bước quá mạnh. Weight decay làm co tham số theo quy tắc của optimizer và có vai trò regularization.
33. **Train và eval ảnh hưởng dropout thế nào?** Train bật bỏ ngẫu nhiên giá trị. Eval tắt dropout. Khi chạy webcam còn dùng inference mode để không xây đồ thị gradient.
34. **Có dùng tuning có hệ thống không?** Chưa có kết quả tìm kiếm có hệ thống. Em nêu cấu hình đã dùng và đề xuất so sánh có kiểm soát trên validation, giữ test cho lần đánh giá cuối.
35. **Tại sao chọn epoch 19 thay vì 20?** Cùng accuracy validation 100%, nhưng epoch 19 có validation CTC loss 0,3325, thấp hơn 0,7207 ở epoch 20.
36. **Kết quả 18/18 nói lên điều gì?** Checkpoint khớp toàn bộ 18 nhãn trên tập test landmark. Tập nhỏ và chưa xác minh độc lập theo người ký, nên chưa suy ra chất lượng webcam.
37. **Precision và recall khác nhau thế nào?** Precision hỏi trong các lần dự đoán nhãn đó có bao nhiêu lần đúng. Recall hỏi trong các mẫu thật của nhãn đó có bao nhiêu được tìm ra. F1 là trung bình điều hòa của hai chỉ số.
38. **Accuracy cao nhưng mô hình yếu ở lớp hiếm có thể xảy ra không?** Có. Đoán “tôi yêu bạn” cho mọi mẫu test hiện tại vẫn đúng 13/18 nhưng recall của “chào” bằng 0. Vì vậy nên xem confusion matrix và macro-F1.
39. **Vì sao accuracy không đổi nhưng validation loss tăng?** Accuracy chỉ xem nhãn sau giải mã. CTC loss còn phụ thuộc xác suất của các đường căn chỉnh hợp lệ. Xác suất đúng có thể giảm trong khi kết quả cuối vẫn đúng.
40. **Có thể kết luận overfitting từ đường cong này không?** Có dấu hiệu ở epoch cuối vì train loss giảm và validation loss tăng. Một mốc chưa đủ kết luận xu hướng dài. Cần tập validation độc lập và chạy lặp để kiểm tra.
41. **Thêm nhãn vào Streamlit có giúp webcam nhận ra nhãn đó không?** Không. Streamlit phát video theo bảng nhãn. Webcam cần dữ liệu gán nhãn, từ vựng và đầu phân loại cập nhật cùng một checkpoint được huấn luyện phù hợp.
42. **Nếu có thêm thời gian, cải tiến nào ưu tiên nhất?** Thu và chia dữ liệu theo người ký và clip nguồn, có cả động tác ngoài từ vựng. Sau đó đo webcam thực tế và đối chứng với baseline trước khi tăng độ phức tạp mô hình.

## 8. Thuật ngữ cần nhớ

| Thuật ngữ | Nghĩa trong dự án |
|---|---|
| Landmark | Tọa độ điểm đặc trưng trên cơ thể hoặc bàn tay |
| Token, vocabulary | ID biểu diễn nhãn và bảng ánh xạ ID sang văn bản |
| Logit, log-xác suất | Điểm thô trước chuẩn hóa và log của xác suất sau chuẩn hóa |
| Batch, epoch | Một nhóm mẫu trong bước cập nhật và một vòng duyệt dữ liệu huấn luyện |
| Parameter, hyperparameter | Trọng số học được và thiết lập do người huấn luyện chọn |
| Self-attention, head | Cơ chế tổng hợp thông tin giữa các khung và một nhóm chiều dùng để tính attention |
| Positional encoding | Thông tin thứ tự khung thêm vào biểu diễn |
| Padding mask | Mặt nạ chặn vị trí đệm trong attention |
| CTC, alignment, blank | Loss học căn chỉnh chuỗi, đường gán token theo khung và token rỗng phục vụ căn chỉnh |
| Greedy decoding | Giải mã dựa vào lựa chọn cục bộ có điểm cao nhất |
| Gradient clipping | Giới hạn chuẩn gradient trước khi cập nhật |
| Label smoothing | Phân bổ một phần trọng số nhãn mục tiêu cho các lớp còn lại trong CE phụ |
| Checkpoint | Gói trọng số, từ vựng, cấu hình và thông tin trạng thái huấn luyện |
| Data leakage | Thông tin từ dữ liệu đánh giá lọt vào quá trình học hoặc chọn mô hình |
| Generalization | Khả năng xử lý dữ liệu mới, đặc biệt từ người ký hoặc điều kiện quay mới |
| CER, WER | Tỉ lệ lỗi theo ký tự và theo từ |
| Macro-F1 | Trung bình F1 theo lớp với trọng số bằng nhau |
| Ablation | So sánh sau khi bỏ hoặc thay một thành phần dưới điều kiện kiểm soát |
| Inference latency | Thời gian từ đầu vào đến lúc có kết quả, cần đo trên thiết bị thật |

## 9. Tệp đối chiếu trong dự án

- Kiến trúc và mã hóa vị trí ở `src/models/transformer.py` và `src/models/positional_encoding.py`.
- Loss, cập nhật trọng số và chọn checkpoint ở `src/training/trainer.py`.
- Dữ liệu và token ở `src/data/dataset.py`, `src/data/tokenizer.py` và `src/features/landmark_extractor.py`.
- Cấu hình, metadata và nhật ký ở `artifacts/chao-toi-yeu-ban/`. Kết quả test ở `artifacts/chao-toi-yeu-ban/reports/metrics.json`.
- Webcam và tra cứu ở `src/inference/realtime.py`, `run_camera.bat` và `streamlit_app.py`.

**Mốc nên thuộc trước buổi thi.** Hai nhãn, 201 chiều, 60 khung, 2 tầng, 4 head, 64 chiều ẩn, 80.260 tham số, 130/15/18 mẫu, 20 epoch, checkpoint 19, 18/18 đúng trên test landmark, chưa đo chính xác webcam.
