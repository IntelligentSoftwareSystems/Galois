#ifndef __ELEMENT_H_INCLUDED__
#define __ELEMENT_H_INCLUDED__

#include "EPosition.hxx"
#include "DoubleArgFunction.hxx"
#include "GaussianQuadrature.hxx"

class Element{
	
	private:
		double xl; 
		double yl; 
		double xr;
		double yr;
		EPosition position;
		bool is_first_tier;
		int bot_left_vertex_nr; 
		int left_edge_nr;
		int top_left_vertex_nr; 
		int top_edge_nr; 
		int top_right_vertex_nr; 
		int bot_edge_nr;
		int interior_nr;
		int right_edge_nr;
		int bot_right_vertex_nr;
		
		IDoubleArgFunction* vertex_bot_left_function;
		IDoubleArgFunction* vertex_top_left_function;
		IDoubleArgFunction* vertex_top_right_function;
		IDoubleArgFunction* vertex_bot_right_function; 
		
		IDoubleArgFunction* edge_left_function;
		IDoubleArgFunction* edge_top_function; 
		IDoubleArgFunction* edge_bot_function; 
		IDoubleArgFunction* edge_right_function; 
		
		IDoubleArgFunction* interior_function;
		
		IDoubleArgFunction** shapeFunctions;
		 
		 
	public:
		Element(double xl, double yl, double xr, double yr, EPosition position, bool is_first_tier)
			: xl(xl), yl(yl), xr(xr), yr(yr), position(position), is_first_tier(is_first_tier)
		{
			
			vertex_bot_left_function = new VertexBotLeftShapeFunction(is_first_tier, xl, yl, xr, yr, position); 
			vertex_top_left_function = new VertexTopLeftShapeFunction(is_first_tier, xl, yl, xr, yr, position); 
			vertex_top_right_function = new VertexTopRightShapeFunction(is_first_tier, xl, yl, xr, yr, position); 
			vertex_bot_right_function = new VertexBotRightShapeFunction(is_first_tier, xl, yl, xr, yr, position);
			
			edge_left_function = new EdgeLeftShapeFunction(is_first_tier, xl, yl, xr, yr, position); 
			edge_top_function = new EdgeTopShapeFunction(is_first_tier, xl, yl, xr, yr, position);
			edge_bot_function = new EdgeBotShapeFunction(is_first_tier, xl, yl, xr, yr, position); 
			edge_right_function = new EdgeRightShapeFunction(is_first_tier, xl, yl, xr, yr, position); 
			
			interior_function = new InteriorShapeFunction(is_first_tier, xl, yl, xr, yr, position); 
			
			
			shapeFunctions = new IDoubleArgFunction*[9];
			shapeFunctions[0] = vertex_bot_left_function;
			shapeFunctions[1] = edge_left_function;
			shapeFunctions[2] = vertex_top_left_function; 
			shapeFunctions[3] = edge_top_function; 
			shapeFunctions[4] = vertex_top_right_function;
			shapeFunctions[5] = edge_bot_function; 
			shapeFunctions[6] = interior_function; 
			shapeFunctions[7] = edge_right_function; 
			shapeFunctions[8] = vertex_bot_right_function;
			
					
		}
		
		~Element()
		{
			delete vertex_bot_left_function;
			delete vertex_top_left_function;
			delete vertex_top_right_function;
			delete vertex_bot_right_function;
			
			delete edge_left_function;
			delete edge_top_function; 
			delete edge_bot_function;
			delete edge_right_function; 
			
			delete interior_function; 
			
			delete shapeFunctions;
			
		}
		
		Element** CreateAnotherTier(int nr);
		Element** CreateFirstTier(int nr);
		Element** CreateLastTier(int nr); 
		
		void fillMatrix(double** matrix);
		void fillRhs(double* rhs, IDoubleArgFunction* f);
		
		void set_bot_left_vertex_nr(int nr){
			bot_left_vertex_nr = nr; 
		}
		void set_top_left_vertex_nr(int nr){
			top_left_vertex_nr = nr; 
		}
		void set_top_right_vertex_nr(int nr){
			top_right_vertex_nr = nr; 
		}
		void set_bot_right_vertex_nr(int nr){
			bot_right_vertex_nr = nr; 
		}
		void set_left_edge_nr(int nr){
			left_edge_nr = nr; 
		}
		void set_top_edge_nr(int nr){
			top_edge_nr = nr; 
		}
		void set_bot_edge_nr(int nr){
			bot_edge_nr = nr; 
		}
		void set_right_edge_nr(int nr){
			right_edge_nr = nr; 
		}
		void set_interior_nr(int nr){
			interior_nr = nr; 
		}
		private:
			void comp(int indx1, int indx2, IDoubleArgFunction* f1, IDoubleArgFunction* f2,double** matrix);
};
#endif
