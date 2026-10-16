# TV2 — Lee & Seung (1999): nền tảng NMF

- **Citation:** Lee, Daniel D. và Seung, H. Sebastian (1999). “Learning the
  parts of objects by non-negative matrix factorization.” *Nature* 401,
  788–791. DOI: [10.1038/44565](https://doi.org/10.1038/44565).
- **Loại:** Bài báo khoa học, nguồn gốc thuật toán NMF.
- **Method:** Phân rã ma trận không âm thành hai nhân tử không âm; biểu diễn
  cộng tính. Bài báo có minh họa đặc trưng ngữ nghĩa của văn bản, nhưng trọng
  tâm thí nghiệm hình ảnh là khuôn mặt.
- **Dataset/metric trong nguồn:** Bài gốc dùng dữ liệu ảnh và minh họa văn bản;
  không dùng Dataset V1 tiếng Việt hoặc NPMI của đồ án.
- **Liên hệ đồ án:** TF-IDF không âm được xấp xỉ bởi W@H; W là trọng số bài–topic,
  H là trọng số topic–từ. Ký hiệu W/H của đồ án theo quy ước hàng là bài, cột là
  từ; bài gốc có thể sắp chiều ma trận khác.
- **Giới hạn/không suy diễn:** Không lấy thí nghiệm ảnh làm bằng chứng NMF tốt hơn
  LDA trên tin tiếng Việt. Hiệu quả trong đồ án phải dựa trên thực nghiệm nhóm.
- **Nơi kiểm tra:** [Trang Nature](https://www.nature.com/articles/44565),
  phần mô tả thuật toán và kết quả; đối chiếu [bản PDF bài gốc](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/lee-nature-99.pdf).
