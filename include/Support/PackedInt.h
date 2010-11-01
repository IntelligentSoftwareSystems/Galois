// Packed Integers into int64 -*- C++ -*-

template<int s1, int s2>
class packedInt2 {
  volatile uint64_t val;
  uint64_t set(unsigned int x, unsigned int y) {
    uint64_t nvx = x;
    nvx = nvx << (64 - s1);
    nvx = nvx >> (64 - s1);
    uint64_t nvy = y;
    nvy = nvy << (64 - s2);
    nvy = nvy >> (64 - s2);
    uint64_t nv = nvx | (nvy << s1);
    return nv;
  }

  void get(uint64_t ov, unsigned int& x, unsigned int& y) {
    uint64_t nx = ov;
    nx = nx << (64 - s1);
    nx = nx >> (64 - s1);
    x = nx;
    uint64_t ny = ov;
    ny = ny >> s1;
    ny = ny << (64 - s2);
    ny = ny >> (64 - s2);
    y = ny;
  }
public:
  packedInt2(int x, int y) {
    val = set(x,y);
  }
  void packedRead(unsigned int& x, unsigned int& y) {
    get(val,x,y);
  }
  void packedWrite(unsigned int x, unsigned int y) {
    val = set(x,y);
  }
  bool CAS(unsigned int oldx, unsigned int oldy, unsigned int newx, unsigned int newy) {
    uint64_t oldv = set(oldx, oldy);
    uint64_t newv = set(newx, newy);
    return __sync_bool_compare_and_swap(&val, oldv, newv);
  }
};


template<int s1, int s2, int s3>
class packedInt3 {
  volatile uint64_t val;
  uint64_t set(unsigned int x, unsigned int y, unsigned int z) {
    uint64_t nvx = x;
    nvx = nvx << (64 - s1);
    nvx = nvx >> (64 - s1);
    uint64_t nvy = y;
    nvy = nvy << (64 - s2);
    nvy = nvy >> (64 - s2);
    uint64_t nvz = z;
    nvz = nvz << (64 - s3);
    nvz = nvz >> (64 - s3);

    uint64_t nv = nvx | (nvy << s1) | (nvz << (s1 + s2));
    return nv;
  }

  void get(uint64_t ov, unsigned int& x, unsigned int& y, unsigned int& z) {
    uint64_t nx = ov;
    nx = nx << (64 - s1);
    nx = nx >> (64 - s1);
    x = nx;
    uint64_t ny = ov;
    ny = ny >> s1;
    ny = ny << (64 - s2);
    ny = ny >> (64 - s2);
    y = ny;
    uint64_t nz = ov;
    nz = nz >> (s1 + s2);
    nz = nz << (64 - s3);
    nz = nz >> (64 - s3);
    z = nz;
  }

public:
  packedInt3(int x, int y, int z) {
    val = set(x,y,z);
  }
  void packedRead(unsigned int& x, unsigned int& y, unsigned int& z) {
    get(val,x,y,z);
  }
  void packedWrite(unsigned int x, unsigned int y, unsigned int z) {
    val = set(x,y,z);
  }
  bool CAS(unsigned int oldx, unsigned int oldy, unsigned int oldz, unsigned int newx, unsigned int newy, unsigned int newz) {
    uint64_t oldv = set(oldx, oldy, oldz);
    uint64_t newv = set(newx, newy, newz);
    return __sync_bool_compare_and_swap(&val, oldv, newv);
  }
};

