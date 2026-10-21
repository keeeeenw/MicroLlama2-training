// Cooperative-tensor layout follows MLX0.32.4 steel/attn/nax.h (Apple Inc., MIT).
// The attention algorithm and kernels here are a separate experimental implementation.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using ffrag=metal::vec<float,8>;
using bfrag=metal::vec<bfloat,8>;

inline short2 nax_coord() {
  ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  return short2{short((((lane >> 2) & 2) | (lane & 1)) * 4),
                short(((lane >> 2) & 4) | ((lane >> 1) & 3))};
}

template <typename T>
inline vec<T,8> frag(const device T* p, int row, int col, int stride,
                     int rows, int cols, bool transpose=false) {
  short2 c=nax_coord(); vec<T,8> result;
  #pragma clang loop unroll(full)
  for(int i=0;i<8;++i) {
    int r=row+c.y+(i/4)*8, s=col+c.x+i%4;
    result[i] = (r<rows && s<cols) ? p[transpose ? s*stride+r : r*stride+s] : T(0);
  }
  return result;
}

inline void mma(thread ffrag& c0, thread ffrag& c1,
                const thread bfrag& a, const thread bfrag& b0, const thread bfrag& b1) {
  constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,false,true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
  auto left=op.template get_left_input_cooperative_tensor<bfloat,bfloat,float>();
  auto right=op.template get_right_input_cooperative_tensor<bfloat,bfloat,float>();
  auto dst=op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>,float>();
  #pragma clang loop unroll(full)
  for(int i=0;i<8;++i) { left[i]=a[i]; right[i]=b0[i]; right[i+8]=b1[i]; dst[i]=c0[i]; dst[i+8]=c1[i]; }
  op.run(left,right,dst);
  #pragma clang loop unroll(full)
  for(int i=0;i<8;++i) { c0[i]=dst[i]; c1[i]=dst[i+8]; }
}

inline float sumrow(float x) { x+=simd_shuffle_xor(x,ushort(1)); return x+simd_shuffle_xor(x,ushort(8)); }
inline float maxrow(float x) { x=max(x,simd_shuffle_xor(x,ushort(1))); return max(x,simd_shuffle_xor(x,ushort(8))); }

inline bfrag tgfrag(const threadgroup bfloat* p,int row,int col,bool transpose) {
  short2 c=nax_coord();bfrag result;
  #pragma clang loop unroll(full)
  for(int i=0;i<8;++i) {
    int r=row+c.y+i/4*8,s=col+c.x+i%4;
    result[i]=p[transpose ? s*64+r : r*64+s];
  }
  return result;
}

template <typename T>
inline void storefrag(device T* p, ffrag value, int row, int col, int length) {
  short2 c=nax_coord();
  #pragma clang loop unroll(full)
  for(int i=0;i<8;++i) {
    int r=row+c.y+(i/4)*8;
    if(r<length) p[r*64+col+c.x+i%4]=T(value[i]);
  }
}
