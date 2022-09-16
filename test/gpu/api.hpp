namespace gpu {

template <typename T>
using device_ptr = T*;

class CollatzMap_1024 {
  static constexpr int BlockN = 1024;
  static constexpr int StreamN = 3;

 private:
  device_ptr<int> block[StreamN];

 public:
  CollatzMap_1024();
  ~CollatzMap_1024();
  void call(const int* in, int* out, int n);
};

}  // namespace gpu