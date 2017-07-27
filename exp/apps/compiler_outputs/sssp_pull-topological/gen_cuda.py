from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("gen_cuda.cuh", system = False)], parse = False),
CDeclGlobal([("unsigned int *", "P_DIST_CURRENT", "")]),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const unsigned int ', 'local_infinity'), ('unsigned int', 'local_src_node'), ('unsigned int *', 'p_dist_current')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_dist_current[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity"]),
]),
]),
]),
Kernel("SSSP", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_dist_current')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("unsigned int", "new_dist", "")]),
CBlock(["new_dist = p_dist_current[dst] + graph.getAbsWeight(jj)"]),
CDecl([("unsigned int", "old_dist", "")]),
CBlock(["old_dist = atomicMin(&p_dist_current[src], new_dist)"]),
If("old_dist > new_dist",
[
ReturnFromParallelFor(" 1"),
]),
]),
),
]),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const unsigned int &', 'local_infinity'), ('unsigned int', 'local_src_node'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "local_infinity", "local_src_node", "ctx->dist_current.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_all_cuda", [('const unsigned int &', 'local_infinity'), ('unsigned int', 'local_src_node'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->nowned, local_infinity, local_src_node, ctx)"]),
], host = True),
Kernel("SSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("SSSP", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->dist_current.data.gpu_wr_ptr()"), "SUM"),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(retval.cpu_rd_ptr())"], parse = False),
], host = True),
Kernel("SSSP_all_cuda", [('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->nowned, __retval, ctx)"]),
], host = True),
])
