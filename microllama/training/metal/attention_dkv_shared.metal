int tile=threadgroup_position_in_grid.x;
int head=tile/((N+63)/64), block=tile%((N+63)/64)*64;
int warp=simdgroup_index_in_threadgroup, local=thread_position_in_threadgroup.x;
int first=block+warp*16;
q+=head*N*64; k+=head*N*64; v+=head*N*64; g+=head*N*64;
dk+=head*N*64; dv+=head*N*64; stats+=head*N*2;
threadgroup bfloat qs[4096],gs[4096],ks[4096],vs[4096];
for(int i=local;i<4096;i+=128) {
  int row=block+i/64;
  ks[i]=row<N ? k[row*64+i%64] : bfloat(0);
  vs[i]=row<N ? v[row*64+i%64] : bfloat(0);
}
short2 c=nax_coord(); ffrag ak[4],av[4];
for(int d=0;d<4;++d) { ak[d]=0.f; av[d]=0.f; }
for(int j=block;j<N;j+=64) {
  for(int i=local;i<4096;i+=128) {
    int row=j+i/64;
    qs[i]=row<N ? q[row*64+i%64] : bfloat(0);
    gs[i]=row<N ? g[row*64+i%64] : bfloat(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for(int chunk=0;chunk<64;chunk+=32) {
    ffrag scores[2]={ffrag(0),ffrag(0)},dp[2]={ffrag(0),ffrag(0)};
    #pragma clang loop unroll(full)
    for(int d=0;d<4;++d) {
      mma(scores[0],scores[1],tgfrag(ks,warp*16,d*16,false),
          tgfrag(qs,d*16,chunk,true),tgfrag(qs,d*16,chunk+16,true));
      mma(dp[0],dp[1],tgfrag(vs,warp*16,d*16,false),
          tgfrag(gs,d*16,chunk,true),tgfrag(gs,d*16,chunk+16,true));
    }
    bfrag ds[2],probabilities[2];
    #pragma clang loop unroll(full)
    for(int b=0;b<2;++b) for(int i=0;i<8;++i) {
      int key=first+c.y+i/4*8,query=j+chunk+b*16+c.x+i%4;
      float p=(key<N && query>=key && query<N) ? exp(scores[b][i]*.125f-stats[query*2]) : 0.f;
      probabilities[b][i]=bfloat(p);
      ds[b][i]=bfloat(p*(dp[b][i]-(query<N ? stats[query*2+1] : 0.f))*.125f);
    }
    #pragma clang loop unroll(full)
    for(int b=0;b<2;++b) for(int d=0;d<4;d+=2) {
      mma(ak[d],ak[d+1],ds[b],tgfrag(qs,chunk+b*16,d*16,false),tgfrag(qs,chunk+b*16,(d+1)*16,false));
      mma(av[d],av[d+1],probabilities[b],tgfrag(gs,chunk+b*16,d*16,false),tgfrag(gs,chunk+b*16,(d+1)*16,false));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(int d=0;d<4;++d) { storefrag(dk,ak[d],first,d*16,N); storefrag(dv,av[d],first,d*16,N); }
