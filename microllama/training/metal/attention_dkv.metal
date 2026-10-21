int tile=threadgroup_position_in_grid.x;
int head=tile/((N+15)/16), first=tile%((N+15)/16)*16;
q+=head*N*64; k+=head*N*64; v+=head*N*64; g+=head*N*64; dk+=head*N*64; dv+=head*N*64; stats+=head*N*2;
short2 c=nax_coord();
bfrag keys[4], values[4]; ffrag ak[4], av[4];
for(int d=0;d<4;++d) { keys[d]=frag(k,first,d*16,64,N,64); values[d]=frag(v,first,d*16,64,N,64); ak[d]=0.f; av[d]=0.f; }
for(int j=first/32*32;j<N;j+=32) {
  ffrag scores[2]={ffrag(0),ffrag(0)}, dp[2]={ffrag(0),ffrag(0)};
  #pragma clang loop unroll(full)
  for(int d=0;d<4;++d) {
    mma(scores[0],scores[1],keys[d],frag(q,d*16,j,64,64,N,true),frag(q,d*16,j+16,64,64,N,true));
    mma(dp[0],dp[1],values[d],frag(g,d*16,j,64,64,N,true),frag(g,d*16,j+16,64,64,N,true));
  }
  bfrag ds[2], probabilities[2];
  for(int b=0;b<2;++b) for(int i=0;i<8;++i) {
    int key=first+c.y+i/4*8, query=j+b*16+c.x+i%4;
    float p=(key<N && query>=key && query<N) ? exp(scores[b][i]*.125f-stats[query*2]) : 0.f;
    probabilities[b][i]=bfloat(p);
    ds[b][i]=bfloat(p*(dp[b][i]-(query<N ? stats[query*2+1] : 0.f))*.125f);
  }
  #pragma clang loop unroll(full)
  for(int b=0;b<2;++b) for(int d=0;d<4;d+=2) {
    mma(ak[d],ak[d+1],ds[b],frag(q,j+b*16,d*16,64,N,64),frag(q,j+b*16,(d+1)*16,64,N,64));
    mma(av[d],av[d+1],probabilities[b],frag(g,j+b*16,d*16,64,N,64),frag(g,j+b*16,(d+1)*16,64,N,64));
  }
}
for(int d=0;d<4;++d) { storefrag(dk,ak[d],first,d*16,N); storefrag(dv,av[d],first,d*16,N); }
