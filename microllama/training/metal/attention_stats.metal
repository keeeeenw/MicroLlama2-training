int tile=threadgroup_position_in_grid.x;
int head=tile/((N+15)/16), first=tile%((N+15)/16)*16;
q+=head*N*64; k+=head*N*64; o+=head*N*64; g+=head*N*64; stats+=head*N*2;
short2 c=nax_coord();
bfrag queries[4];
float2 delta=0.f, largest=-INFINITY, total=0.f;
#pragma clang loop unroll(full)
for(int d=0;d<4;++d) {
  queries[d]=frag(q,first,d*16,64,N,64);
  auto a=frag(o,first,d*16,64,N,64), b=frag(g,first,d*16,64,N,64);
  for(int i=0;i<8;++i) delta[i/4]+=float(a[i])*float(b[i]);
}
for(int j=0;j<min(N,first+16);j+=32) {
  ffrag scores[2]={ffrag(0),ffrag(0)};
  #pragma clang loop unroll(full)
  for(int d=0;d<4;++d) mma(scores[0],scores[1],queries[d],
      frag(k,d*16,j,64,64,N,true),frag(k,d*16,j+16,64,64,N,true));
  for(int r=0;r<2;++r) {
    float hi=-INFINITY;
    for(int b=0;b<2;++b) for(int s=0;s<4;++s) {
      int query=first+c.y+r*8, key=j+b*16+c.x+s;
      float x=key<=query && key<N ? scores[b][r*4+s]*.125f : -INFINITY;
      scores[b][r*4+s]=x; hi=max(hi,x);
    }
    hi=maxrow(hi); float next=max(largest[r],hi), amount=0.f;
    for(int b=0;b<2;++b) for(int s=0;s<4;++s) amount+=exp(scores[b][r*4+s]-next);
    total[r]=total[r]*exp(largest[r]-next)+sumrow(amount); largest[r]=next;
  }
}
for(int r=0;r<2;++r) {
  float dot=sumrow(delta[r]); int row=first+c.y+r*8;
  if(c.x==0 && row<N) { stats[row*2]=largest[r]+log(total[r]); stats[row*2+1]=dot; }
}
