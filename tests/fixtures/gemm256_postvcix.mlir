#map = affine_map<(d0, d1) -> (d0 * 256 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 65536 + d1 * 256)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0, d1) -> (d0 * 256 + d1 * 512)>
#map4 = affine_map<(d0, d1, d2) -> (-d0 + d1 + d2 floordiv 2)>
#map5 = affine_map<(d0, d1, d2)[s0, s1] -> (d0 * s0 + d1 * s1 + d2)>
#map6 = affine_map<(d0)[s0] -> (d0 floordiv s0)>
#map7 = affine_map<(d0)[s0] -> (d0 mod s0)>
#map8 = affine_map<(d0, d1, d2) -> (-d0 + d1 * 2 + d2)>
module {
  memref.global @X_spad : memref<256x256xf32, 1>
  memref.global @W_spad : memref<256x256xf32, 1>
  memref.global @Y_spad : memref<256x256xf32, 1>
  func.func @kernel(%arg0: memref<65536xf32>, %arg1: memref<65536xf32>, %arg2: memref<65536xf32>) {
    %0 = memref.get_global @X_spad : memref<256x256xf32, 1>
    %1 = memref.get_global @W_spad : memref<256x256xf32, 1>
    %2 = memref.get_global @Y_spad : memref<256x256xf32, 1>
    %cst = arith.constant dense<0.000000e+00> : vector<512xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<1xi32>
    affine.for %arg3 = 0 to 256 step 256 {
      affine.for %arg4 = 0 to 256 step 256 {
        affine.vector_store %cst, %2[0, 0] : memref<256x256xf32, 1>, vector<512xf32>
        affine.for %arg5 = 0 to 256 step 256 {
          %4 = affine.apply #map(%arg3, %arg5)
          %c1_1 = arith.constant 1 : index
          %alloc_2 = memref.alloc() : memref<1xi32>
          %5 = affine.apply #map(%arg5, %arg4)
          %c1_3 = arith.constant 1 : index
          %alloc_4 = memref.alloc() : memref<1xi32>
          %c0_5 = arith.constant 0 : index
          %c0_6 = arith.constant 0 : index
          %c0_7 = arith.constant 0 : index
          %6 = affine.apply #map1(%c0_5, %c0_6)
          %7 = affine.apply #map2(%6, %4)
          %8 = affine.apply #map3(%c0_5, %c0_6)
          %9 = affine.apply #map2(%c0_5, %c0_6)
          memref.dma_start %arg0[%7], %0[%c0_7, %8], %c2, %alloc_2[%9], %c1_1, %c1 : memref<65536xf32>, memref<256x256xf32, 1>, memref<1xi32> {async = true, dram_stride = [256, 1], fine_grained = true, sram_stride = [1, 256], subtile_size = [256, 256]}
          %c0_8 = arith.constant 0 : index
          %c0_9 = arith.constant 0 : index
          %c0_10 = arith.constant 0 : index
          %10 = affine.apply #map1(%c0_8, %c0_9)
          %11 = affine.apply #map2(%10, %5)
          %12 = affine.apply #map3(%c0_8, %c0_9)
          %13 = affine.apply #map2(%c0_8, %c0_9)
          memref.dma_start %arg1[%11], %1[%c0_10, %12], %c2, %alloc_4[%13], %c1_3, %c1 : memref<65536xf32>, memref<256x256xf32, 1>, memref<1xi32> {async = true, dram_stride = [256, 1], fine_grained = true, sram_stride = [1, 256], subtile_size = [256, 256]}
          %c0_11 = arith.constant 0 : index
          %c8_i64 = arith.constant 8 : i64
          %c256 = arith.constant 256 : index
          %c256_12 = arith.constant 256 : index
          %c256_13 = arith.constant 256 : index
          %c128 = arith.constant 128 : index
          %c1_14 = arith.constant 1 : index
          %cst_15 = arith.constant 0.000000e+00 : f32
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %14 = affine.apply #map4(%arg5, %c0_11, %c0_11)
              memref.dma_wait %alloc_4[%14], %c1_14 : memref<1xi32>
              %c0_16 = arith.constant 0 : index
              %c128_17 = arith.constant 128 : index
              %15 = affine.apply #map5(%arg6, %arg7, %c0_16)[%c256, %c128_17]
              %16 = affine.apply #map6(%15)[%c256_12]
              %17 = affine.apply #map7(%15)[%c256_12]
              %18 = vector.transfer_read %1[%16, %17], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%18, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c8 = arith.constant 8 : index
              %c128_18 = arith.constant 128 : index
              %19 = affine.apply #map5(%arg6, %arg7, %c8)[%c256, %c128_18]
              %20 = affine.apply #map6(%19)[%c256_12]
              %21 = affine.apply #map7(%19)[%c256_12]
              %22 = vector.transfer_read %1[%20, %21], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%22, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c16 = arith.constant 16 : index
              %c128_19 = arith.constant 128 : index
              %23 = affine.apply #map5(%arg6, %arg7, %c16)[%c256, %c128_19]
              %24 = affine.apply #map6(%23)[%c256_12]
              %25 = affine.apply #map7(%23)[%c256_12]
              %26 = vector.transfer_read %1[%24, %25], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%26, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c24 = arith.constant 24 : index
              %c128_20 = arith.constant 128 : index
              %27 = affine.apply #map5(%arg6, %arg7, %c24)[%c256, %c128_20]
              %28 = affine.apply #map6(%27)[%c256_12]
              %29 = affine.apply #map7(%27)[%c256_12]
              %30 = vector.transfer_read %1[%28, %29], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%30, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c32 = arith.constant 32 : index
              %c128_21 = arith.constant 128 : index
              %31 = affine.apply #map5(%arg6, %arg7, %c32)[%c256, %c128_21]
              %32 = affine.apply #map6(%31)[%c256_12]
              %33 = affine.apply #map7(%31)[%c256_12]
              %34 = vector.transfer_read %1[%32, %33], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%34, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c40 = arith.constant 40 : index
              %c128_22 = arith.constant 128 : index
              %35 = affine.apply #map5(%arg6, %arg7, %c40)[%c256, %c128_22]
              %36 = affine.apply #map6(%35)[%c256_12]
              %37 = affine.apply #map7(%35)[%c256_12]
              %38 = vector.transfer_read %1[%36, %37], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%38, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c48 = arith.constant 48 : index
              %c128_23 = arith.constant 128 : index
              %39 = affine.apply #map5(%arg6, %arg7, %c48)[%c256, %c128_23]
              %40 = affine.apply #map6(%39)[%c256_12]
              %41 = affine.apply #map7(%39)[%c256_12]
              %42 = vector.transfer_read %1[%40, %41], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%42, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c56 = arith.constant 56 : index
              %c128_24 = arith.constant 128 : index
              %43 = affine.apply #map5(%arg6, %arg7, %c56)[%c256, %c128_24]
              %44 = affine.apply #map6(%43)[%c256_12]
              %45 = affine.apply #map7(%43)[%c256_12]
              %46 = vector.transfer_read %1[%44, %45], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%46, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c64 = arith.constant 64 : index
              %c128_25 = arith.constant 128 : index
              %47 = affine.apply #map5(%arg6, %arg7, %c64)[%c256, %c128_25]
              %48 = affine.apply #map6(%47)[%c256_12]
              %49 = affine.apply #map7(%47)[%c256_12]
              %50 = vector.transfer_read %1[%48, %49], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%50, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c72 = arith.constant 72 : index
              %c128_26 = arith.constant 128 : index
              %51 = affine.apply #map5(%arg6, %arg7, %c72)[%c256, %c128_26]
              %52 = affine.apply #map6(%51)[%c256_12]
              %53 = affine.apply #map7(%51)[%c256_12]
              %54 = vector.transfer_read %1[%52, %53], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%54, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c80 = arith.constant 80 : index
              %c128_27 = arith.constant 128 : index
              %55 = affine.apply #map5(%arg6, %arg7, %c80)[%c256, %c128_27]
              %56 = affine.apply #map6(%55)[%c256_12]
              %57 = affine.apply #map7(%55)[%c256_12]
              %58 = vector.transfer_read %1[%56, %57], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%58, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c88 = arith.constant 88 : index
              %c128_28 = arith.constant 128 : index
              %59 = affine.apply #map5(%arg6, %arg7, %c88)[%c256, %c128_28]
              %60 = affine.apply #map6(%59)[%c256_12]
              %61 = affine.apply #map7(%59)[%c256_12]
              %62 = vector.transfer_read %1[%60, %61], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%62, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c96 = arith.constant 96 : index
              %c128_29 = arith.constant 128 : index
              %63 = affine.apply #map5(%arg6, %arg7, %c96)[%c256, %c128_29]
              %64 = affine.apply #map6(%63)[%c256_12]
              %65 = affine.apply #map7(%63)[%c256_12]
              %66 = vector.transfer_read %1[%64, %65], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%66, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c104 = arith.constant 104 : index
              %c128_30 = arith.constant 128 : index
              %67 = affine.apply #map5(%arg6, %arg7, %c104)[%c256, %c128_30]
              %68 = affine.apply #map6(%67)[%c256_12]
              %69 = affine.apply #map7(%67)[%c256_12]
              %70 = vector.transfer_read %1[%68, %69], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%70, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c112 = arith.constant 112 : index
              %c128_31 = arith.constant 128 : index
              %71 = affine.apply #map5(%arg6, %arg7, %c112)[%c256, %c128_31]
              %72 = affine.apply #map6(%71)[%c256_12]
              %73 = affine.apply #map7(%71)[%c256_12]
              %74 = vector.transfer_read %1[%72, %73], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%74, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              %c120 = arith.constant 120 : index
              %c128_32 = arith.constant 128 : index
              %75 = affine.apply #map5(%arg6, %arg7, %c120)[%c256, %c128_32]
              %76 = affine.apply #map6(%75)[%c256_12]
              %77 = affine.apply #map7(%75)[%c256_12]
              %78 = vector.transfer_read %1[%76, %77], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
              "vcix.iv"(%78, %c8_i64) {imm = 0 : i64, opcode = 1 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
              affine.for %arg8 = 0 to 2 {
                %79 = affine.apply #map8(%arg5, %c0_11, %c0_11)
                memref.dma_wait %alloc_2[%79], %c1_14 : memref<1xi32>
                %c0_33 = arith.constant 0 : index
                %80 = affine.apply #map5(%arg7, %arg8, %c0_33)[%c256_13, %c128]
                %81 = affine.apply #map6(%80)[%c256]
                %82 = affine.apply #map7(%80)[%c256]
                %83 = vector.transfer_read %0[%81, %82], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%83, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c8_34 = arith.constant 8 : index
                %84 = affine.apply #map5(%arg7, %arg8, %c8_34)[%c256_13, %c128]
                %85 = affine.apply #map6(%84)[%c256]
                %86 = affine.apply #map7(%84)[%c256]
                %87 = vector.transfer_read %0[%85, %86], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%87, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c16_35 = arith.constant 16 : index
                %88 = affine.apply #map5(%arg7, %arg8, %c16_35)[%c256_13, %c128]
                %89 = affine.apply #map6(%88)[%c256]
                %90 = affine.apply #map7(%88)[%c256]
                %91 = vector.transfer_read %0[%89, %90], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%91, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c24_36 = arith.constant 24 : index
                %92 = affine.apply #map5(%arg7, %arg8, %c24_36)[%c256_13, %c128]
                %93 = affine.apply #map6(%92)[%c256]
                %94 = affine.apply #map7(%92)[%c256]
                %95 = vector.transfer_read %0[%93, %94], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%95, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c32_37 = arith.constant 32 : index
                %96 = affine.apply #map5(%arg7, %arg8, %c32_37)[%c256_13, %c128]
                %97 = affine.apply #map6(%96)[%c256]
                %98 = affine.apply #map7(%96)[%c256]
                %99 = vector.transfer_read %0[%97, %98], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%99, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c40_38 = arith.constant 40 : index
                %100 = affine.apply #map5(%arg7, %arg8, %c40_38)[%c256_13, %c128]
                %101 = affine.apply #map6(%100)[%c256]
                %102 = affine.apply #map7(%100)[%c256]
                %103 = vector.transfer_read %0[%101, %102], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%103, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c48_39 = arith.constant 48 : index
                %104 = affine.apply #map5(%arg7, %arg8, %c48_39)[%c256_13, %c128]
                %105 = affine.apply #map6(%104)[%c256]
                %106 = affine.apply #map7(%104)[%c256]
                %107 = vector.transfer_read %0[%105, %106], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%107, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c56_40 = arith.constant 56 : index
                %108 = affine.apply #map5(%arg7, %arg8, %c56_40)[%c256_13, %c128]
                %109 = affine.apply #map6(%108)[%c256]
                %110 = affine.apply #map7(%108)[%c256]
                %111 = vector.transfer_read %0[%109, %110], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%111, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c64_41 = arith.constant 64 : index
                %112 = affine.apply #map5(%arg7, %arg8, %c64_41)[%c256_13, %c128]
                %113 = affine.apply #map6(%112)[%c256]
                %114 = affine.apply #map7(%112)[%c256]
                %115 = vector.transfer_read %0[%113, %114], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%115, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c72_42 = arith.constant 72 : index
                %116 = affine.apply #map5(%arg7, %arg8, %c72_42)[%c256_13, %c128]
                %117 = affine.apply #map6(%116)[%c256]
                %118 = affine.apply #map7(%116)[%c256]
                %119 = vector.transfer_read %0[%117, %118], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%119, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c80_43 = arith.constant 80 : index
                %120 = affine.apply #map5(%arg7, %arg8, %c80_43)[%c256_13, %c128]
                %121 = affine.apply #map6(%120)[%c256]
                %122 = affine.apply #map7(%120)[%c256]
                %123 = vector.transfer_read %0[%121, %122], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%123, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c88_44 = arith.constant 88 : index
                %124 = affine.apply #map5(%arg7, %arg8, %c88_44)[%c256_13, %c128]
                %125 = affine.apply #map6(%124)[%c256]
                %126 = affine.apply #map7(%124)[%c256]
                %127 = vector.transfer_read %0[%125, %126], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%127, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c96_45 = arith.constant 96 : index
                %128 = affine.apply #map5(%arg7, %arg8, %c96_45)[%c256_13, %c128]
                %129 = affine.apply #map6(%128)[%c256]
                %130 = affine.apply #map7(%128)[%c256]
                %131 = vector.transfer_read %0[%129, %130], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%131, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c104_46 = arith.constant 104 : index
                %132 = affine.apply #map5(%arg7, %arg8, %c104_46)[%c256_13, %c128]
                %133 = affine.apply #map6(%132)[%c256]
                %134 = affine.apply #map7(%132)[%c256]
                %135 = vector.transfer_read %0[%133, %134], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%135, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c112_47 = arith.constant 112 : index
                %136 = affine.apply #map5(%arg7, %arg8, %c112_47)[%c256_13, %c128]
                %137 = affine.apply #map6(%136)[%c256]
                %138 = affine.apply #map7(%136)[%c256]
                %139 = vector.transfer_read %0[%137, %138], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%139, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                %c120_48 = arith.constant 120 : index
                %140 = affine.apply #map5(%arg7, %arg8, %c120_48)[%c256_13, %c128]
                %141 = affine.apply #map6(%140)[%c256]
                %142 = affine.apply #map7(%140)[%c256]
                %143 = vector.transfer_read %0[%141, %142], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                "vcix.iv"(%143, %c8_i64) {imm = 0 : i64, opcode = 0 : i64, rd = 0 : i64} : (vector<8xf32>, i64) -> ()
                "vcix.i"(%c8_i64) {imm = 4 : i64, lmul = 0 : i64, opcode = 1 : i64, rd = 0 : i64, rs2 = 0 : i64, sew = 32 : i64} : (i64) -> ()
                %c0_49 = arith.constant 0 : index
                %144 = affine.apply #map5(%arg6, %arg8, %c0_49)[%c256_13, %c128]
                %145 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %146 = affine.apply #map6(%144)[%c256_12]
                %147 = affine.apply #map7(%144)[%c256_12]
                %148 = vector.transfer_read %2[%146, %147], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %149 = arith.addf %148, %145 : vector<8xf32>
                vector.transfer_write %149, %2[%146, %147] : vector<8xf32>, memref<256x256xf32, 1>
                %c8_50 = arith.constant 8 : index
                %150 = affine.apply #map5(%arg6, %arg8, %c8_50)[%c256_13, %c128]
                %151 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %152 = affine.apply #map6(%150)[%c256_12]
                %153 = affine.apply #map7(%150)[%c256_12]
                %154 = vector.transfer_read %2[%152, %153], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %155 = arith.addf %154, %151 : vector<8xf32>
                vector.transfer_write %155, %2[%152, %153] : vector<8xf32>, memref<256x256xf32, 1>
                %c16_51 = arith.constant 16 : index
                %156 = affine.apply #map5(%arg6, %arg8, %c16_51)[%c256_13, %c128]
                %157 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %158 = affine.apply #map6(%156)[%c256_12]
                %159 = affine.apply #map7(%156)[%c256_12]
                %160 = vector.transfer_read %2[%158, %159], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %161 = arith.addf %160, %157 : vector<8xf32>
                vector.transfer_write %161, %2[%158, %159] : vector<8xf32>, memref<256x256xf32, 1>
                %c24_52 = arith.constant 24 : index
                %162 = affine.apply #map5(%arg6, %arg8, %c24_52)[%c256_13, %c128]
                %163 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %164 = affine.apply #map6(%162)[%c256_12]
                %165 = affine.apply #map7(%162)[%c256_12]
                %166 = vector.transfer_read %2[%164, %165], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %167 = arith.addf %166, %163 : vector<8xf32>
                vector.transfer_write %167, %2[%164, %165] : vector<8xf32>, memref<256x256xf32, 1>
                %c32_53 = arith.constant 32 : index
                %168 = affine.apply #map5(%arg6, %arg8, %c32_53)[%c256_13, %c128]
                %169 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %170 = affine.apply #map6(%168)[%c256_12]
                %171 = affine.apply #map7(%168)[%c256_12]
                %172 = vector.transfer_read %2[%170, %171], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %173 = arith.addf %172, %169 : vector<8xf32>
                vector.transfer_write %173, %2[%170, %171] : vector<8xf32>, memref<256x256xf32, 1>
                %c40_54 = arith.constant 40 : index
                %174 = affine.apply #map5(%arg6, %arg8, %c40_54)[%c256_13, %c128]
                %175 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %176 = affine.apply #map6(%174)[%c256_12]
                %177 = affine.apply #map7(%174)[%c256_12]
                %178 = vector.transfer_read %2[%176, %177], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %179 = arith.addf %178, %175 : vector<8xf32>
                vector.transfer_write %179, %2[%176, %177] : vector<8xf32>, memref<256x256xf32, 1>
                %c48_55 = arith.constant 48 : index
                %180 = affine.apply #map5(%arg6, %arg8, %c48_55)[%c256_13, %c128]
                %181 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %182 = affine.apply #map6(%180)[%c256_12]
                %183 = affine.apply #map7(%180)[%c256_12]
                %184 = vector.transfer_read %2[%182, %183], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %185 = arith.addf %184, %181 : vector<8xf32>
                vector.transfer_write %185, %2[%182, %183] : vector<8xf32>, memref<256x256xf32, 1>
                %c56_56 = arith.constant 56 : index
                %186 = affine.apply #map5(%arg6, %arg8, %c56_56)[%c256_13, %c128]
                %187 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %188 = affine.apply #map6(%186)[%c256_12]
                %189 = affine.apply #map7(%186)[%c256_12]
                %190 = vector.transfer_read %2[%188, %189], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %191 = arith.addf %190, %187 : vector<8xf32>
                vector.transfer_write %191, %2[%188, %189] : vector<8xf32>, memref<256x256xf32, 1>
                %c64_57 = arith.constant 64 : index
                %192 = affine.apply #map5(%arg6, %arg8, %c64_57)[%c256_13, %c128]
                %193 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %194 = affine.apply #map6(%192)[%c256_12]
                %195 = affine.apply #map7(%192)[%c256_12]
                %196 = vector.transfer_read %2[%194, %195], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %197 = arith.addf %196, %193 : vector<8xf32>
                vector.transfer_write %197, %2[%194, %195] : vector<8xf32>, memref<256x256xf32, 1>
                %c72_58 = arith.constant 72 : index
                %198 = affine.apply #map5(%arg6, %arg8, %c72_58)[%c256_13, %c128]
                %199 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %200 = affine.apply #map6(%198)[%c256_12]
                %201 = affine.apply #map7(%198)[%c256_12]
                %202 = vector.transfer_read %2[%200, %201], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %203 = arith.addf %202, %199 : vector<8xf32>
                vector.transfer_write %203, %2[%200, %201] : vector<8xf32>, memref<256x256xf32, 1>
                %c80_59 = arith.constant 80 : index
                %204 = affine.apply #map5(%arg6, %arg8, %c80_59)[%c256_13, %c128]
                %205 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %206 = affine.apply #map6(%204)[%c256_12]
                %207 = affine.apply #map7(%204)[%c256_12]
                %208 = vector.transfer_read %2[%206, %207], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %209 = arith.addf %208, %205 : vector<8xf32>
                vector.transfer_write %209, %2[%206, %207] : vector<8xf32>, memref<256x256xf32, 1>
                %c88_60 = arith.constant 88 : index
                %210 = affine.apply #map5(%arg6, %arg8, %c88_60)[%c256_13, %c128]
                %211 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %212 = affine.apply #map6(%210)[%c256_12]
                %213 = affine.apply #map7(%210)[%c256_12]
                %214 = vector.transfer_read %2[%212, %213], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %215 = arith.addf %214, %211 : vector<8xf32>
                vector.transfer_write %215, %2[%212, %213] : vector<8xf32>, memref<256x256xf32, 1>
                %c96_61 = arith.constant 96 : index
                %216 = affine.apply #map5(%arg6, %arg8, %c96_61)[%c256_13, %c128]
                %217 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %218 = affine.apply #map6(%216)[%c256_12]
                %219 = affine.apply #map7(%216)[%c256_12]
                %220 = vector.transfer_read %2[%218, %219], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %221 = arith.addf %220, %217 : vector<8xf32>
                vector.transfer_write %221, %2[%218, %219] : vector<8xf32>, memref<256x256xf32, 1>
                %c104_62 = arith.constant 104 : index
                %222 = affine.apply #map5(%arg6, %arg8, %c104_62)[%c256_13, %c128]
                %223 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %224 = affine.apply #map6(%222)[%c256_12]
                %225 = affine.apply #map7(%222)[%c256_12]
                %226 = vector.transfer_read %2[%224, %225], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %227 = arith.addf %226, %223 : vector<8xf32>
                vector.transfer_write %227, %2[%224, %225] : vector<8xf32>, memref<256x256xf32, 1>
                %c112_63 = arith.constant 112 : index
                %228 = affine.apply #map5(%arg6, %arg8, %c112_63)[%c256_13, %c128]
                %229 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %230 = affine.apply #map6(%228)[%c256_12]
                %231 = affine.apply #map7(%228)[%c256_12]
                %232 = vector.transfer_read %2[%230, %231], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %233 = arith.addf %232, %229 : vector<8xf32>
                vector.transfer_write %233, %2[%230, %231] : vector<8xf32>, memref<256x256xf32, 1>
                %c120_64 = arith.constant 120 : index
                %234 = affine.apply #map5(%arg6, %arg8, %c120_64)[%c256_13, %c128]
                %235 = "vcix.v.i"(%c8_i64) {imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64} : (i64) -> vector<8xf32>
                %236 = affine.apply #map6(%234)[%c256_12]
                %237 = affine.apply #map7(%234)[%c256_12]
                %238 = vector.transfer_read %2[%236, %237], %cst_15 : memref<256x256xf32, 1>, vector<8xf32>
                %239 = arith.addf %238, %235 : vector<8xf32>
                vector.transfer_write %239, %2[%236, %237] : vector<8xf32>, memref<256x256xf32, 1>
              } {inner_loop = true}
            } {inner_loop = true}
          } {inner_loop = true}
        } {accumulation_loop = true, subtile_loop = "k"}
        affine.for %arg5 = 0 to 1 {
        } {inner_loop = false}
        %3 = affine.apply #map(%arg3, %arg4)
        %c1_0 = arith.constant 1 : index
        memref.dma_start %2[%c0, %c0], %arg2[%3], %c3, %alloc[%c0], %c1_0, %c1 : memref<256x256xf32, 1>, memref<65536xf32>, memref<1xi32> {dram_stride = [256, 1], padding = 0 : i64, sram_stride = [1, 256]}
      } {outer_loop = true, subtile_loop = "n"}
    } {outer_loop = true, subtile_loop = "m"}
    return
  }
  func.func @wrapper_kernel(%arg0: memref<65536xf32>, %arg1: memref<65536xf32>, %arg2: memref<65536xf32>) {
    call @kernel(%arg0, %arg1, %arg2) : (memref<65536xf32>, memref<65536xf32>, memref<65536xf32>) -> ()
    return
  }
}
