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
CDeclGlobal([("unsigned int *", "P_COMP_CURRENT", "")]),
Kernel("InitializeGraph", [G.param(), ('unsigned int', 'nowned') , ('unsigned int *', 'p_comp_current')],
[
ForAll("src", G.nodes(None, "nowned"),
[
CDecl([("bool", "pop", " = src < nowned")]),
If("pop", [
CBlock(["p_comp_current[src] = graph.node_data[src]"]),
]),
]),
]),
Kernel("ConnectedComp", [G.param(), ('unsigned int', 'nowned') , ('unsigned int *', 'p_comp_current'), ('Any', 'any_retval')],
[
ForAll("src", G.nodes(None, "nowned"),
[
CDecl([("bool", "pop", " = src < nowned")]),
If("pop", [
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("unsigned int", "new_comp", "")]),
CBlock(["new_comp = p_comp_current[dst]"]),
CDecl([("unsigned int", "old_comp", "")]),
CBlock(["old_comp = atomicMin(&p_comp_current[src], new_comp)"]),
If("old_comp > new_comp",
[
CBlock(["any_retval.return_( 1)"]),
]),
]),
),
]),
]),
Kernel("InitializeGraph_cuda", [('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(ctx->gg, blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "ctx->comp_current.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("ConnectedComp_cuda", [('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(ctx->gg, blocks, threads)"]),
CBlock(["*(ctx->p_retval.cpu_wr_ptr()) = __retval"]),
CBlock(["ctx->any_retval.rv = ctx->p_retval.gpu_wr_ptr()"]),
Invoke("ConnectedComp", ("ctx->gg", "ctx->nowned", "ctx->comp_current.gpu_wr_ptr()", "ctx->any_retval")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(ctx->p_retval.cpu_rd_ptr())"]),
], host = True),
])
