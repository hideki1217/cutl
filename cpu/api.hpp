namespace cpu {

class CollatzMap {
 public:
  CollatzMap();
  ~CollatzMap();
  void call(const int* in, int* out, int n);
};

}  // namespace cpu