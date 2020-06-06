from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("tc_cuda.cuh", system = False)], parse = False),
CBlock([cgen.Include("kernels/segmentedsort.cuh", system = False)], parse = False),
CBlock([cgen.Include("moderngpu.cuh", system = False)], parse = False),
CBlock([cgen.Include("util/mgpucontext.h", system = False)], parse = False),
CBlock([cgen.Include("cuda_profiler_api.h", system = True)], parse = False),

CBlock('mgpu::ContextPtr mgc', parse = False),

Kernel("intersect", [G.param(), ('index_type', 'u'), ('index_type', 'v')],
       [
        CDecl([('index_type', 'u_start', '= graph.getFirstEdge(u)'),
               ('index_type', 'u_end', '= u_start + graph.getOutDegree(u)'),
               ('index_type', 'v_start', '= graph.getFirstEdge(v)'),
               ('index_type', 'v_end', '= v_start + graph.getOutDegree(v)'),
               ('int', 'count', '= 0'),
               ('index_type', 'u_it', '= u_start'),
               ('index_type', 'v_it', '= v_start'),
               ('index_type', 'a', ''),
               ('index_type', 'b', ''),                       
               ]),
        While('u_it < u_end && v_it < v_end',
              [
                CBlock('a = graph.getAbsDestination(u_it)'),
                CBlock('b = graph.getAbsDestination(v_it)'),                        
                CDecl(('int', 'd', '= a - b')),
                If('d <= 0', [CBlock('u_it++')]),
                If('d >= 0', [CBlock('v_it++')]),
                If('d == 0', [CBlock('count++')]),
                ]
              ),                      
        CBlock('return count'),
        ],
       device=True,
       ret_type = 'unsigned int',
),

Kernel("TC", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('HGAccumulator<unsigned int>', 'num_local_triangles')],
       [CDecl([("__shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage", "num_local_triangles_ts", "")]),
		CBlock(["num_local_triangles.thread_entry()"]),
		ForAll("src", G.nodes("__begin", "__end"),
               [CDecl([("bool", "pop", " = src < __end")]),
				UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
                ClosureHint(ForAll("edge", G.edges("src"), 
                                   [CDecl([('index_type', 'u', '= graph.getAbsDestination(edge)'),
                                           ('index_type', 'd_u', '= graph.getOutDegree(u)'),
                                           ('int', 'xcount', '= 0')]),
                                    CBlock('xcount = intersect(graph, u, src)'),
                                    If('xcount', [CBlock(["num_local_triangles.reduce(xcount)"])])
                                    ]
                                   )
                            ),
                ]
        ),
		CBlock(["num_local_triangles.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(num_local_triangles_ts)"], parse = False),
        ],
),

Kernel("TC_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('unsigned int &', 'num_local_triangles'), ('struct CUDA_Context* ', 'ctx')],
[
	CDecl([("dim3", "blocks", "")]),
	CDecl([("dim3", "threads", "")]),
	CBlock(["kernel_sizing(blocks, threads)"]),
	CDecl([("Shared<unsigned int>", "num_local_trianglesval", " = Shared<unsigned int>(1)")]),
	CDecl([("HGAccumulator<unsigned int>", "_num_local_triangles", "")]),
	CBlock(["*(num_local_trianglesval.cpu_wr_ptr()) = 0"]),
	CBlock(["_num_local_triangles.rv = num_local_trianglesval.gpu_wr_ptr()"]),
	CBlock(["mgc = mgpu::CreateCudaDevice(ctx->device)"], parse=False),
	CBlock("mgpu::SegSortKeysFromIndices(ctx->gg.edge_dst, ctx->gg.nedges, (const int *) ctx->gg.row_start + 1, ctx->gg.nnodes - 1, *mgc)", parse=False),
	Invoke("TC", ("ctx->gg", "__begin", "__end",  "_num_local_triangles")),
	CBlock(["check_cuda_kernel"], parse = False),
	CBlock(["num_local_triangles = *(num_local_trianglesval.cpu_rd_ptr())"]),
	CBlock('dump_memory_info("end", ctx->id)', parse=False),
	CBlock('cudaProfilerStop()', parse=False),
], host = True),

Kernel("TC_masterNodes_cuda", [('unsigned int &', 'num_local_triangles'), ('struct CUDA_Context* ', 'ctx')],
[
	CBlock(["TC_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, num_local_triangles, ctx)"]),
], host = True),
])

