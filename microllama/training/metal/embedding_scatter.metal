uint first=threadgroup_position_in_grid.x;
uint token=positions[first];
auto row=ids[token];
if(first>0 && ids[positions[first-1]]==row) return;
uint local=thread_position_in_threadgroup.x;
for(uint dim=local;dim<D;dim+=256) {
  float sum=0.f;
  for(uint i=first;i<L && ids[positions[i]]==row;++i)
    sum=sum+cotangent[positions[i]*D+dim];
  gradient[uint(row)*D+dim]=sum;
}
