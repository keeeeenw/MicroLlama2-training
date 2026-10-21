int tile=threadgroup_position_in_grid.x;
int head=tile/((N+15)/16), first=tile%((N+15)/16)*16;
q+=head*N*64; k+=head*N*64; v+=head*N*64; g+=head*N*64; dq+=head*N*64; stats+=head*N*2;
short2 c=nax_coord();
bfrag queries[4], upstream[4]; ffrag accum[4];
for(int d=0;d<4;++d) { queries[d]=frag(q,first,d*16,64,N,64); upstream[d]=frag(g,first,d*16,64,N,64); accum[d]=0.f; }
for(int j=0;j<min(N,first+16);j+=32) {
  ffrag scores[2]={ffrag(0),ffrag(0)}, dp[2]={ffrag(0),ffrag(0)};
  #pragma clang loop unroll(full)
  for(int d=0;d<4;++d) {
    mma(scores[0],scores[1],queries[d],frag(k,d*16,j,64,64,N,true),frag(k,d*16,j+16,64,64,N,true));
    mma(dp[0],dp[1],upstream[d],frag(v,d*16,j,64,64,N,true),frag(v,d*16,j+16,64,64,N,true));
  }
  bfrag ds[2];
  for(int b=0;b<2;++b) for(int i=0;i<8;++i) {
    int query=first+c.y+i/4*8, key=j+b*16+c.x+i%4;
    float p=(query<N && key<=query && key<N) ? exp(scores[b][i]*.125f-stats[query*2]) : 0.f;
    ds[b][i]=bfloat(p*(dp[b][i]-(query<N ? stats[query*2+1] : 0.f))*.125f);
  }
  #pragma clang loop unroll(full)
  for(int b=0;b<2;++b) for(int d=0;d<4;d+=2)
    mma(accum[d],accum[d+1],ds[b],frag(k,j+b*16,d*16,64,N,64),frag(k,j+b*16,(d+1)*16,64,N,64));
}
for(int d=0;d<4;++d) storefrag(dq,accum[d],first,d*16,N);
