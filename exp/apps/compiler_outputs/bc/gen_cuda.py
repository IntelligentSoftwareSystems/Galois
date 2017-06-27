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
CDeclGlobal([("float *", "P_BETWEENESS_CENTRALITY", "")]),
CDeclGlobal([("unsigned int *", "P_CURRENT_LENGTH", "")]),
CDeclGlobal([("float *", "P_DEPENDENCY", "")]),
CDeclGlobal([("unsigned int *", "P_NUM_PREDECESSORS", "")]),
CDeclGlobal([("unsigned int *", "P_NUM_SHORTEST_PATHS", "")]),
CDeclGlobal([("unsigned int *", "P_NUM_SUCCESSORS", "")]),
CDeclGlobal([("unsigned int *", "P_OLD_LENGTH", "")]),
CDeclGlobal([("bool *", "P_PROPOGATION_FLAG", "")]),
CDeclGlobal([("unsigned int *", "P_TRIM", "")]),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('float *', 'p_betweeness_centrality')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_betweeness_centrality[src] = 0"]),
]),
]),
]),
Kernel("InitializeIteration", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const unsigned int ', 'local_current_src_node'), ('const unsigned int ', 'local_infinity'), ('unsigned int *', 'p_current_length'), ('float *', 'p_dependency'), ('unsigned int *', 'p_num_predecessors'), ('unsigned int *', 'p_num_shortest_paths'), ('unsigned int *', 'p_num_successors'), ('unsigned int *', 'p_old_length'), ('bool *', 'p_propogation_flag'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_current_length[src] = (graph.node_data[src] == local_current_src_node) ?  0 : local_infinity"]),
CBlock(["p_old_length[src] = (graph.node_data[src] == local_current_src_node) ?  0 : local_infinity"]),
CBlock(["p_trim[src] = 0"]),
CBlock(["p_propogation_flag[src] = false"]),
CBlock(["p_num_shortest_paths[src] = (graph.node_data[src] == local_current_src_node) ?  1 : 0"]),
CBlock(["p_num_successors[src] = 0"]),
CBlock(["p_num_predecessors[src] = 0"]),
CBlock(["p_dependency[src] = 0"]),
]),
]),
]),
Kernel("FirstIterationSSSP", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_current_length')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("unsigned int", "new_dist", "")]),
CBlock(["new_dist = 1 + p_current_length[src]"]),
CBlock(["atomicMin(&p_current_length[dst], new_dist)"]),
]),
),
]),
]),
Kernel("SSSP", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_current_length'), ('unsigned int *', 'p_old_length')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_old_length[src] > p_current_length[src]",
[
CBlock(["p_old_length[src] = p_current_length[src]"]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("unsigned int", "new_dist", "")]),
CBlock(["new_dist = 1 + p_current_length[src]"]),
CBlock(["atomicMin(&p_current_length[dst], new_dist)"]),
]),
),
ReturnFromParallelFor(" 1"),
]),
]),
Kernel("PredAndSucc", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_current_length'), ('unsigned int *', 'p_num_predecessors'), ('unsigned int *', 'p_num_successors')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("unsigned int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
If("(p_current_length[src] + edge_weight) == p_current_length[dst]",
[
CBlock(["atomicAdd(&p_num_successors[src], (unsigned int)1)"]),
CBlock(["atomicAdd(&p_num_predecessors[dst], (unsigned int)1)"]),
]),
]),
),
]),
]),
Kernel("PredecessorDecrement", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_num_predecessors'), ('bool *', 'p_propogation_flag'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_trim[src] > 0",
[
If("p_trim[src] > p_num_predecessors[src]",
[
CBlock(["abort()"]),
]),
CBlock(["p_num_predecessors[src] -= p_trim[src]"]),
CBlock(["p_trim[src] = 0"]),
If("p_num_predecessors[src] == 0",
[
CBlock(["p_propogation_flag[src] = false"]),
]),
]),
]),
]),
]),
Kernel("NumShortestPaths", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_current_length'), ('unsigned int *', 'p_num_predecessors'), ('unsigned int *', 'p_num_shortest_paths'), ('bool *', 'p_propogation_flag'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_num_predecessors[src] == 0 && !p_propogation_flag[src]",
[
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("unsigned int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
CDecl([("unsigned int", "to_add", "")]),
CBlock(["to_add = p_num_shortest_paths[src]"]),
If("(p_current_length[src] + edge_weight) == p_current_length[dst]",
[
CBlock(["atomicAdd(&p_num_shortest_paths[dst], to_add)"]),
CBlock(["atomicAdd(&p_trim[dst], (unsigned int)1)"]),
ReturnFromParallelFor(" 1"),
]),
]),
),
CBlock(["p_propogation_flag[src] = true"]),
]),
]),
Kernel("PropFlagReset", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('bool *', 'p_propogation_flag')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_propogation_flag[src] = false"]),
]),
]),
]),
Kernel("SuccessorDecrement", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_num_successors'), ('bool *', 'p_propogation_flag'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_num_successors[src] == 0 && !p_propogation_flag[src]",
[
CBlock(["p_propogation_flag[src] = true"]),
If("p_trim[src] > 0",
[
If("p_trim[src] > p_num_successors[src]",
[
CBlock(["abort()"]),
]),
CBlock(["p_num_successors[src] -= p_trim[src]"]),
CBlock(["p_trim[src] = 0"]),
If("p_num_successors[src] == 0",
[
CBlock(["p_propogation_flag[src] = false"]),
]),
]),
]),
]),
]),
]),
Kernel("DependencyPropogation", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const unsigned int ', 'local_current_src_node'), ('unsigned int *', 'p_current_length'), ('float *', 'p_dependency'), ('unsigned int *', 'p_num_shortest_paths'), ('unsigned int *', 'p_num_successors'), ('bool *', 'p_propogation_flag'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("graph.node_data[src] == local_current_src_node || p_num_successors[src] == 0",
[
]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("unsigned int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
If("p_num_successors[dst] == 0 && !p_propogation_flag[dst]",
[
If("(p_current_length[src] + edge_weight) == p_current_length[dst]",
[
CBlock(["atomicAdd(&p_trim[src], (unsigned int)1)"]),
CBlock(["p_dependency[src] = p_dependency[src] + (((float)p_num_shortest_paths[src] / (float)p_num_shortest_paths[dst]) * (float)(1.0 + p_dependency[dst]))"]),
ReturnFromParallelFor(" 1"),
]),
]),
]),
),
]),
]),
Kernel("BC", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('float *', 'p_betweeness_centrality'), ('float *', 'p_dependency')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_betweeness_centrality[src] = p_betweeness_centrality[src] + p_dependency[src]"]),
]),
]),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->betweeness_centrality.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("InitializeIteration_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const unsigned int &', 'local_infinity'), ('const unsigned int &', 'local_current_src_node'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeIteration", ("ctx->gg", "ctx->nowned", "__begin", "__end", "local_current_src_node", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->old_length.data.gpu_wr_ptr()", "ctx->propogation_flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeIteration_all_cuda", [('const unsigned int &', 'local_infinity'), ('const unsigned int &', 'local_current_src_node'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeIteration_cuda(0, ctx->nowned, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("FirstIterationSSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("FirstIterationSSSP", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->current_length.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("FirstIterationSSSP_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["FirstIterationSSSP_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("SSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("SSSP", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->current_length.data.gpu_wr_ptr()", "ctx->old_length.data.gpu_wr_ptr()"), "SUM"),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(retval.cpu_rd_ptr())"], parse = False),
], host = True),
Kernel("SSSP_all_cuda", [('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->nowned, __retval, ctx)"]),
], host = True),
Kernel("PredAndSucc_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("PredAndSucc", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->current_length.data.gpu_wr_ptr()", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("PredAndSucc_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["PredAndSucc_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("PredecessorDecrement_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("PredecessorDecrement", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->propogation_flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("PredecessorDecrement_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["PredecessorDecrement_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("NumShortestPaths_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("NumShortestPaths", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->current_length.data.gpu_wr_ptr()", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->propogation_flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()"), "SUM"),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(retval.cpu_rd_ptr())"], parse = False),
], host = True),
Kernel("NumShortestPaths_all_cuda", [('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["NumShortestPaths_cuda(0, ctx->nowned, __retval, ctx)"]),
], host = True),
Kernel("PropFlagReset_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("PropFlagReset", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->propogation_flag.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("PropFlagReset_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["PropFlagReset_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("SuccessorDecrement_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("SuccessorDecrement", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->propogation_flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("SuccessorDecrement_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["SuccessorDecrement_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("DependencyPropogation_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('int &', '__retval'), ('const unsigned int &', 'local_current_src_node'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("DependencyPropogation", ("ctx->gg", "ctx->nowned", "__begin", "__end", "local_current_src_node", "ctx->current_length.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->propogation_flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()"), "SUM"),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(retval.cpu_rd_ptr())"], parse = False),
], host = True),
Kernel("DependencyPropogation_all_cuda", [('int &', '__retval'), ('const unsigned int &', 'local_current_src_node'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["DependencyPropogation_cuda(0, ctx->nowned, __retval, local_current_src_node, ctx)"]),
], host = True),
Kernel("BC_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("BC", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->betweeness_centrality.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("BC_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["BC_cuda(0, ctx->nowned, ctx)"]),
], host = True),
])
