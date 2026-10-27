module attributes {pytorchsim.copy_node = {args = [0], why = "arg 0: loaded whole once at the kernel's top, the copy node; arg 1: loaded whole too, read as loaded (graph_prefetch is (0,))"}, pytorchsim.widths = "{\22dot_lmul\22: {}, \22lmul\22: 8, \22loop_lmul\22: {}}", "ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @pytorchsim_triton_compiler_fused_mm_au7umwrt(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c32768 = arith.constant 32768 : index
    %c3 = arith.constant 3 : index
    %c32768_0 = arith.constant 32768 : index
    %c2 = arith.constant 2 : index
    %c65536 = arith.constant 65536 : index
    %c1 = arith.constant 1 : index
    %c65536_1 = arith.constant 65536 : index
    %c0 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c32_i64 = arith.constant 32 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_3 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = arith.muli %c0_2, %c64 : index
    %c128 = arith.constant 128 : index
    %1 = arith.muli %c0_2, %c128 : index
    %2 = arith.muli %0, %c1_3 : index
    %3 = arith.addi %1, %2 : index
    %c32 = arith.constant 32 : index
    %4 = arith.addi %3, %c32 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256xbf16, 1>
    %5 = "pytorchsim_tpu.vpu.bank_view"(%alloc) : (memref<256x256xbf16, 1>) -> memref<1x512xbf16, strided<[512, 1]>, 1>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x256xf32, 1>
    %6 = "pytorchsim_tpu.vpu.bank_view"(%alloc_4) : (memref<256x256xf32, 1>) -> memref<1x512xf32, strided<[512, 1]>, 1>
    %alloc_5 = memref.alloc() {alignment = 64 : i64, pytorchsim.prefetch_operand = 65536 : i64} : memref<256x256xbf16, 1>
    %7 = "pytorchsim_tpu.vpu.bank_view"(%alloc_5) : (memref<256x256xbf16, 1>) -> memref<1x512xbf16, strided<[512, 1]>, 1>
    %alloc_6 = memref.alloc() {alignment = 64 : i64, pytorchsim.prefetch_operand = 65536 : i64} : memref<256x256xbf16, 1>
    %8 = "pytorchsim_tpu.vpu.bank_view"(%alloc_6) : (memref<256x256xbf16, 1>) -> memref<1x512xbf16, strided<[512, 1]>, 1>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [65536], strides: [1] : memref<*xbf16> to memref<65536xbf16>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [65536], strides: [1] : memref<*xbf16> to memref<65536xbf16>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [65536], strides: [1] : memref<*xbf16> to memref<65536xbf16>
    "pytorchsim.transfer"(%reinterpret_cast_8, %c0_2, %alloc_6, %c0_2, %c0) <{async = true, direction = "load", dram_arg = 0 : i64, dram_stride = [1, 256], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : (memref<65536xbf16>, index, memref<256x256xbf16, 1>, index, index) -> ()
    "pytorchsim.transfer"(%reinterpret_cast_7, %c0_2, %alloc_5, %c0_2, %c1) <{async = true, direction = "load", dram_arg = 1 : i64, dram_stride = [1, 256], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : (memref<65536xbf16>, index, memref<256x256xbf16, 1>, index, index) -> ()
    %9 = "pytorchsim_tpu.vpu.bank_words"(%8) : (memref<1x512xbf16, strided<[512, 1]>, 1>) -> memref<1x256xi32, strided<[256, 1]>, 1>
    %10 = "pytorchsim_tpu.vpu.bank_words"(%7) : (memref<1x512xbf16, strided<[512, 1]>, 1>) -> memref<1x256xi32, strided<[256, 1]>, 1>
    %11 = "pytorchsim_tpu.vpu.bank_words"(%5) : (memref<1x512xbf16, strided<[512, 1]>, 1>) -> memref<1x256xi32, strided<[256, 1]>, 1>
    "pytorchsim.wait"(%c1, %c65536, %alloc_5) : (index, index, memref<256x256xbf16, 1>) -> ()
    %12 = vector.transfer_read %10[%c0_2, %4], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 0 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%12, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 0 : i64} : (vector<32xi32>, i64) -> ()
    %13 = arith.addi %3, %c0_2 : index
    %14 = vector.transfer_read %10[%c0_2, %13], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 0 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%14, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 0 : i64} : (vector<32xi32>, i64) -> ()
    %15 = arith.muli %c0_2, %c32 : index
    %16 = arith.addi %1, %15 : index
    %17 = arith.addi %16, %c0_2 : index
    "pytorchsim.wait"(%c0, %c65536_1, %alloc_6) : (index, index, memref<256x256xbf16, 1>) -> ()
    %18 = vector.transfer_read %9[%c0_2, %17], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%18, %c32_i64) <{imm = 7 : i64, opcode = 0 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %19 = arith.addi %c0_2, %c1_3 : index
    %20 = arith.muli %19, %c64 : index
    %21 = arith.muli %20, %c1_3 : index
    %22 = arith.addi %1, %21 : index
    %23 = arith.addi %22, %c32 : index
    %24 = vector.transfer_read %10[%c0_2, %23], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%24, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %25 = arith.muli %c1_3, %c32 : index
    %26 = arith.addi %1, %25 : index
    %27 = arith.addi %26, %c0_2 : index
    %28 = vector.transfer_read %9[%c0_2, %27], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%28, %c32_i64) <{imm = 3 : i64, opcode = 0 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %29 = arith.addi %22, %c0_2 : index
    %30 = vector.transfer_read %10[%c0_2, %29], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%30, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %c2_9 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %31 = arith.muli %c0_2, %c256 : index
    %32 = arith.addi %31, %0 : index
    %33 = arith.addi %32, %c0_2 : index
    %34 = arith.addi %32, %c32 : index
    %35 = arith.muli %c1_3, %c128 : index
    %36 = arith.addi %35, %15 : index
    %37 = arith.addi %36, %c0_2 : index
    affine.for %arg9 = 0 to 2 {
      %89 = arith.addi %arg9, %c2_9 : index
      %90 = arith.muli %89, %c32 : index
      %91 = arith.addi %1, %90 : index
      %92 = arith.addi %91, %c0_2 : index
      %93 = vector.transfer_read %9[%c0_2, %92], %c0_i32 {in_bounds = [true]} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
      "vcix.iv"(%93, %c32_i64) <{imm = 3 : i64, opcode = 0 : i64, rd = 0 : i64}> : (vector<32xi32>, i64) -> ()
    } {pytorchsim.tile_id = 1 : i64}
    %38 = vector.transfer_read %9[%c0_2, %37], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    %39 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (i64) -> vector<32xi32>
    %40 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (i64) -> vector<32xi32>
    %41 = arith.bitcast %39 {pytorchsim.tile_id = 1 : i64} : vector<32xi32> to vector<32xf32>
    %42 = arith.bitcast %40 {pytorchsim.tile_id = 1 : i64} : vector<32xi32> to vector<32xf32>
    vector.transfer_write %41, %6[%c0_2, %33] {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
    vector.transfer_write %42, %6[%c0_2, %34] {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
    "vcix.iv"(%38, %c32_i64) <{imm = 7 : i64, opcode = 0 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %43 = arith.addi %35, %2 : index
    %44 = arith.addi %43, %c32 : index
    %45 = vector.transfer_read %10[%c0_2, %44], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%45, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %46 = arith.muli %c1_3, %c64 : index
    %47 = arith.addi %31, %46 : index
    %48 = arith.addi %47, %c0_2 : index
    %49 = arith.addi %47, %c32 : index
    %50 = arith.addi %35, %25 : index
    %51 = arith.addi %50, %c0_2 : index
    %52 = vector.transfer_read %9[%c0_2, %51], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    %53 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (i64) -> vector<32xi32>
    %54 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (i64) -> vector<32xi32>
    %55 = arith.bitcast %53 {pytorchsim.tile_id = 1 : i64} : vector<32xi32> to vector<32xf32>
    %56 = arith.bitcast %54 {pytorchsim.tile_id = 1 : i64} : vector<32xi32> to vector<32xf32>
    vector.transfer_write %55, %6[%c0_2, %48] {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
    vector.transfer_write %56, %6[%c0_2, %49] {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
    "vcix.iv"(%52, %c32_i64) <{imm = 3 : i64, opcode = 0 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %57 = arith.addi %43, %c0_2 : index
    %58 = vector.transfer_read %10[%c0_2, %57], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%58, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg9 = 0 to 2 {
      %89 = arith.addi %arg9, %c2_9 : index
      %90 = arith.muli %89, %c64 : index
      %91 = arith.addi %31, %90 : index
      %92 = arith.addi %91, %c0_2 : index
      %93 = arith.addi %92, %c32 : index
      %94 = arith.muli %89, %c32 : index
      %95 = arith.addi %35, %94 : index
      %96 = arith.addi %95, %c0_2 : index
      %97 = vector.transfer_read %9[%c0_2, %96], %c0_i32 {in_bounds = [true]} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
      %98 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %99 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %100 = arith.bitcast %98 : vector<32xi32> to vector<32xf32>
      %101 = arith.bitcast %99 : vector<32xi32> to vector<32xf32>
      vector.transfer_write %100, %6[%c0_2, %92] {in_bounds = [true]} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
      vector.transfer_write %101, %6[%c0_2, %93] {in_bounds = [true]} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
      "vcix.iv"(%97, %c32_i64) <{imm = 3 : i64, opcode = 0 : i64, rd = 0 : i64}> : (vector<32xi32>, i64) -> ()
    } {pytorchsim.tile_id = 1 : i64}
    %59 = vector.transfer_read %6[%c0_2, %33], %cst {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x512xf32, strided<[512, 1]>, 1>, vector<32xf32>
    %60 = vector.transfer_read %6[%c0_2, %34], %cst {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x512xf32, strided<[512, 1]>, 1>, vector<32xf32>
    %61 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (i64) -> vector<32xi32>
    %62 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (i64) -> vector<32xi32>
    %63 = arith.bitcast %61 {pytorchsim.tile_id = 1 : i64} : vector<32xi32> to vector<32xf32>
    %64 = arith.bitcast %62 {pytorchsim.tile_id = 1 : i64} : vector<32xi32> to vector<32xf32>
    %65 = arith.addf %59, %63 {pytorchsim.tile_id = 1 : i64} : vector<32xf32>
    %66 = arith.addf %60, %64 {pytorchsim.tile_id = 1 : i64} : vector<32xf32>
    %67 = arith.bitcast %65 {pytorchsim.tile_id = 1 : i64} : vector<32xf32> to vector<32xi32>
    %68 = arith.bitcast %66 {pytorchsim.tile_id = 1 : i64} : vector<32xf32> to vector<32xi32>
    %69 = "vcix.v.sv"(%67, %68) <{opcode = 2 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    %70 = vector.transfer_read %9[%c0_2, %17], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    vector.transfer_write %69, %11[%c0_2, %17] {in_bounds = [true], pytorchsim.tile_id = 1 : i64} : vector<32xi32>, memref<1x256xi32, strided<[256, 1]>, 1>
    "vcix.iv"(%70, %c32_i64) <{imm = 7 : i64, opcode = 0 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 1 : i64} : (vector<32xi32>, i64) -> ()
    %71 = arith.addi %c0_2, %c0_2 : index
    %72 = arith.muli %46, %c1_3 : index
    %73 = arith.addi %35, %72 : index
    %74 = arith.addi %73, %c32 : index
    affine.for %arg9 = 0 to 3 {
      %89 = arith.addi %arg9, %c1_3 : index
      %90 = arith.muli %89, %c64 : index
      %91 = arith.addi %31, %90 : index
      %92 = arith.addi %91, %c0_2 : index
      %93 = arith.addi %92, %c32 : index
      %94 = vector.transfer_read %6[%c0_2, %92], %cst {in_bounds = [true]} : memref<1x512xf32, strided<[512, 1]>, 1>, vector<32xf32>
      %95 = vector.transfer_read %6[%c0_2, %93], %cst {in_bounds = [true]} : memref<1x512xf32, strided<[512, 1]>, 1>, vector<32xf32>
      %96 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %97 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %98 = arith.bitcast %96 : vector<32xi32> to vector<32xf32>
      %99 = arith.bitcast %97 : vector<32xi32> to vector<32xf32>
      %100 = arith.addf %94, %98 : vector<32xf32>
      %101 = arith.addf %95, %99 : vector<32xf32>
      %102 = arith.bitcast %100 : vector<32xf32> to vector<32xi32>
      %103 = arith.bitcast %101 : vector<32xf32> to vector<32xi32>
      %104 = "vcix.v.sv"(%102, %103) <{opcode = 2 : i64}> : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
      %105 = arith.muli %89, %c32 : index
      %106 = arith.addi %1, %105 : index
      %107 = arith.addi %106, %c0_2 : index
      vector.transfer_write %104, %11[%c0_2, %107] {in_bounds = [true]} : vector<32xi32>, memref<1x256xi32, strided<[256, 1]>, 1>
      %108 = vector.transfer_read %9[%c0_2, %107], %c0_i32 {in_bounds = [true]} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
      "vcix.iv"(%108, %c32_i64) <{imm = 3 : i64, opcode = 0 : i64, rd = 0 : i64}> : (vector<32xi32>, i64) -> ()
    } {pytorchsim.tile_id = 1 : i64}
    "pytorchsim.transfer"(%reinterpret_cast, %71, %alloc, %71, %c2) <{async = true, direction = "store", dram_arg = 2 : i64, dram_stride = [1, 256], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>, subtile_size = [128, 256]}> : (memref<65536xbf16>, index, memref<256x256xbf16, 1>, index, index) -> ()
    %75 = vector.transfer_read %10[%c0_2, %74], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 2 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%75, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 2 : i64} : (vector<32xi32>, i64) -> ()
    %76 = arith.addi %73, %c0_2 : index
    %77 = vector.transfer_read %10[%c0_2, %76], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 2 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    "vcix.iv"(%77, %c32_i64) <{imm = 11 : i64, opcode = 1 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 2 : i64} : (vector<32xi32>, i64) -> ()
    %78 = arith.muli %c1_3, %c256 : index
    %79 = arith.addi %78, %0 : index
    %80 = arith.addi %79, %c0_2 : index
    %81 = arith.addi %79, %c32 : index
    %82 = vector.transfer_read %9[%c0_2, %37], %c0_i32 {in_bounds = [true], pytorchsim.tile_id = 2 : i64} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
    %83 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 2 : i64} : (i64) -> vector<32xi32>
    %84 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> {pytorchsim.tile_id = 2 : i64} : (i64) -> vector<32xi32>
    %85 = arith.bitcast %83 {pytorchsim.tile_id = 2 : i64} : vector<32xi32> to vector<32xf32>
    %86 = arith.bitcast %84 {pytorchsim.tile_id = 2 : i64} : vector<32xi32> to vector<32xf32>
    vector.transfer_write %85, %6[%c0_2, %80] {in_bounds = [true], pytorchsim.tile_id = 2 : i64} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
    vector.transfer_write %86, %6[%c0_2, %81] {in_bounds = [true], pytorchsim.tile_id = 2 : i64} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
    "vcix.iv"(%82, %c32_i64) <{imm = 7 : i64, opcode = 0 : i64, rd = 0 : i64}> {pytorchsim.tile_id = 2 : i64} : (vector<32xi32>, i64) -> ()
    %87 = arith.addi %c0_2, %c128 : index
    %88 = arith.addi %c0_2, %c256 : index
    affine.for %arg9 = 0 to 3 {
      %89 = arith.addi %arg9, %c1_3 : index
      %90 = arith.muli %89, %c64 : index
      %91 = arith.addi %78, %90 : index
      %92 = arith.addi %91, %c0_2 : index
      %93 = arith.addi %92, %c32 : index
      %94 = arith.muli %89, %c32 : index
      %95 = arith.addi %35, %94 : index
      %96 = arith.addi %95, %c0_2 : index
      %97 = vector.transfer_read %9[%c0_2, %96], %c0_i32 {in_bounds = [true]} : memref<1x256xi32, strided<[256, 1]>, 1>, vector<32xi32>
      %98 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %99 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %100 = arith.bitcast %98 : vector<32xi32> to vector<32xf32>
      %101 = arith.bitcast %99 : vector<32xi32> to vector<32xf32>
      vector.transfer_write %100, %6[%c0_2, %92] {in_bounds = [true]} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
      vector.transfer_write %101, %6[%c0_2, %93] {in_bounds = [true]} : vector<32xf32>, memref<1x512xf32, strided<[512, 1]>, 1>
      "vcix.iv"(%97, %c32_i64) <{imm = 3 : i64, opcode = 0 : i64, rd = 0 : i64}> : (vector<32xi32>, i64) -> ()
    } {pytorchsim.tile_id = 2 : i64}
    "pytorchsim.wait"(%c2, %c32768_0, %alloc) : (index, index, memref<256x256xbf16, 1>) -> ()
    affine.for %arg9 = 0 to 4 {
      %89 = arith.muli %arg9, %c64 : index
      %90 = arith.addi %78, %89 : index
      %91 = arith.addi %90, %c0_2 : index
      %92 = arith.addi %91, %c32 : index
      %93 = vector.transfer_read %6[%c0_2, %91], %cst {in_bounds = [true]} : memref<1x512xf32, strided<[512, 1]>, 1>, vector<32xf32>
      %94 = vector.transfer_read %6[%c0_2, %92], %cst {in_bounds = [true]} : memref<1x512xf32, strided<[512, 1]>, 1>, vector<32xf32>
      %95 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %96 = "vcix.v.i"(%c32_i64) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<32xi32>
      %97 = arith.bitcast %95 : vector<32xi32> to vector<32xf32>
      %98 = arith.bitcast %96 : vector<32xi32> to vector<32xf32>
      %99 = arith.addf %93, %97 : vector<32xf32>
      %100 = arith.addf %94, %98 : vector<32xf32>
      %101 = arith.bitcast %99 : vector<32xf32> to vector<32xi32>
      %102 = arith.bitcast %100 : vector<32xf32> to vector<32xi32>
      %103 = "vcix.v.sv"(%101, %102) <{opcode = 2 : i64}> : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
      %104 = arith.muli %arg9, %c32 : index
      %105 = arith.addi %35, %104 : index
      %106 = arith.addi %105, %c0_2 : index
      vector.transfer_write %103, %11[%c0_2, %106] {in_bounds = [true]} : vector<32xi32>, memref<1x256xi32, strided<[256, 1]>, 1>
    } {pytorchsim.tile_id = 3 : i64}
    "pytorchsim.transfer"(%reinterpret_cast, %87, %alloc, %88, %c3) <{async = true, direction = "store", dram_arg = 2 : i64, dram_stride = [1, 256], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>, subtile_size = [128, 256]}> : (memref<65536xbf16>, index, memref<256x256xbf16, 1>, index, index) -> ()
    "pytorchsim.wait"(%c3, %c32768, %alloc) : (index, index, memref<256x256xbf16, 1>) -> ()
    memref.dealloc %alloc : memref<256x256xbf16, 1>
    memref.dealloc %alloc_4 : memref<256x256xf32, 1>
    memref.dealloc %alloc_5 : memref<256x256xbf16, 1>
    memref.dealloc %alloc_6 : memref<256x256xbf16, 1>
    return
  }
}
