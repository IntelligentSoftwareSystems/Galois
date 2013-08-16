#ifndef __ELEMENT_H_INCLUDED__
#define __ELEMENT_H_INCLUDED__

#include "EPosition.hxx"
#include "TripleArgFunction.hxx"
#include "GaussianQuadrature.hxx"
#include <stdio.h>
namespace tmp{
class Element{
	
	private:
		double xl; 
		double yl; 
		double xr;
		double yr;
		double zl;
		double zr;
		double size;
		EPosition position;
		bool is_first_tier;
		
		static const int nr_of_nodes = 27;
		

		ITripleArgFunction** shapeFunctions;
		int* shapeFunctionNrs;

		void SetIternalBotInterfaceNumbers(int nr, Element* bot_left_near_element, Element* bot_left_far_element, Element* bot_right_far_element,
				Element* top_left_near_element, Element* top_left_far_element, Element* top_right_far_elemnt, Element* top_right_near_element);

	public:

		static const int vertex_bot_left_near = 0;
		static const int edge_bot_left = 1;
		static const int vertex_bot_left_far = 2;
		static const int edge_bot_far = 3;
		static const int vertex_bot_right_far = 4;
		static const int edge_left_near = 5;
		static const int face_left = 6;
		static const int edge_left_far = 7;
		static const int face_far = 8;
		static const int edge_right_far = 9;
		static const int vertex_top_left_near = 10;
		static const int edge_top_left = 11;
		static const int vertex_top_left_far = 12;
		static const int edge_top_far = 13;
		static const int vertex_top_right_far = 14;
		static const int edge_bot_near = 15;
		static const int face_bot = 16;
		static const int edge_bot_right = 17;
		static const int face_near = 18;
		static const int interior = 19;
		static const int face_right = 20;
		static const int edge_top_near = 21;
		static const int face_top = 22;
		static const int edge_top_right = 23;
		static const int vertex_bot_right_near = 24;
		static const int edge_right_near = 25;
		static const int vertex_top_right_near = 26;
		
		Element(double xl, double yl, double zl, double size,  EPosition position, bool is_first_tier)
			: xl(xl), yl(yl), zl(zl), size(size), position(position), is_first_tier(is_first_tier)
		{

			xr = xl + size;
			yr = yl + size;
			zr = zl + size;
			shapeFunctions = new ITripleArgFunction*[nr_of_nodes];
			shapeFunctionNrs = new int[nr_of_nodes];
			
			shapeFunctions[vertex_bot_left_near] = new VertexBotLeftNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[vertex_bot_left_far] = new VertexBotLeftFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[vertex_bot_right_near] = new VertexBotRightNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[vertex_bot_right_far] = new VertexBotRightFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[vertex_top_left_near] = new VertexTopLeftNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[vertex_top_left_far] = new VertexTopLeftFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[vertex_top_right_near] = new VertexTopRightNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[vertex_top_right_far] = new VertexTopRightFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			
			shapeFunctions[edge_bot_left] = new EdgeBotLeftShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_bot_right] = new EdgeBotRightShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_bot_near] = new EdgeBotNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_bot_far] = new EdgeBotFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_top_left] = new EdgeTopLeftShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_top_right] = new EdgeTopRightShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_top_near] = new EdgeTopNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_top_far] = new EdgeTopFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_left_near] = new EdgeLeftNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_left_far] = new EdgeLeftFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_right_near] = new EdgeRightNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[edge_right_far] = new EdgeRightFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			
			shapeFunctions[face_left] = new FaceLeftShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[face_right] = new FaceRightShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[face_top] = new FaceTopShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[face_bot] = new FaceBotShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[face_near] = new FaceNearShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			shapeFunctions[face_far] = new FaceFarShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
			
			shapeFunctions[interior] = new InteriorShapeFunction(is_first_tier,xl,yl,zl,xr,yr,zr,position);
		}
		
		~Element()
		{
			delete[] shapeFunctionNrs;
			for(int i = 0; i<nr_of_nodes; i++)
				delete shapeFunctions[i];
			
			delete[] shapeFunctions;
			
		}
		
		Element** CreateAnotherTier(int nr);
		Element** CreateFirstTier(int nr);
		Element** CreateLastTier(int nr);
		
		void fillMatrix(double** matrix);
		void fillMatrix(double** matrix, int start_nr_adj);
		void fillRhs(double* rhs, ITripleArgFunction* f);
		void fillRhs(double* rhs, ITripleArgFunction* f, int start_nr_adj);
		void fillTierMatrix(double** matrix, double* rhs, ITripleArgFunction* f, int start_nr_adj);
		
		void set_node_nr(int node, int node_nr)
		{
			shapeFunctionNrs[node] = node_nr;
		}
		int get_node_nr(int node)
		{
			return shapeFunctionNrs[node];
		}
		int get_bot_left_near_vertex_nr()
		{
			return shapeFunctionNrs[vertex_bot_left_near];
		}
		private:
			void comp(int indx1, int indx2, ITripleArgFunction* f1, ITripleArgFunction* f2,double** matrix);
};
}
#endif
