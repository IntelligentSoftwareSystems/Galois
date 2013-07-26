#ifndef __DOUBLEARGFUNCTION_H_INCLUDED
#define __DOUBLEARGFUNCTION_H_INCLUDED
#include "EPosition.hxx"

class IDoubleArgFunction
{
	
	public:
		virtual double ComputeValue(double x, double y) = 0; 
};


class ShapeFunction : public IDoubleArgFunction
{
	protected :
		double xl; 
		double yl; 
		double xr; 
		double yr;
		bool is_first_tier; 
		EPosition position; 
		
	
		double getXValueOnElement(double x)
		{
			
			return (x-xl)/(xr-xl); 
		}
		
		double getYValueOnElement(double y)
		{
			return (y-yl)/(yr-yl); 
		}
		
	
	public:
		ShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			:  xl(xl), yl(yl), xr(xr), yr(yr), is_first_tier(is_first_tier), position(position)
		{
			
		}
		
		~ShapeFunction()
		{
		}
		
};


class VertexBotLeftShapeFunction : public ShapeFunction
{
	public:

		VertexBotLeftShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		
		~VertexBotLeftShapeFunction()
		{
		}
	
};

class VertexTopLeftShapeFunction : public ShapeFunction
{
	public:

		VertexTopLeftShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~VertexTopLeftShapeFunction()
		{
		}
};

class VertexTopRightShapeFunction : public ShapeFunction
{
	public:

		VertexTopRightShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~VertexTopRightShapeFunction()
		{
		}
};

class VertexBotRightShapeFunction : public ShapeFunction
{
	public:

		VertexBotRightShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~VertexBotRightShapeFunction()
		{
		}
};

class EdgeLeftShapeFunction : public ShapeFunction
{
	public:
	
		EdgeLeftShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
				: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
			{
				
			}
			
		virtual double ComputeValue(double x, double y);
		~EdgeLeftShapeFunction()
		{
		}
};

class EdgeTopShapeFunction : public ShapeFunction
{
	public:

		EdgeTopShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~EdgeTopShapeFunction()
		{
		}
};

class EdgeBotShapeFunction : public ShapeFunction
{
	public:

		EdgeBotShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~EdgeBotShapeFunction()
		{
		}
};

class EdgeRightShapeFunction : public ShapeFunction
{	
	public:

		EdgeRightShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~EdgeRightShapeFunction()
		{
		}
};

class InteriorShapeFunction : public ShapeFunction
{
	public:

		InteriorShapeFunction(bool is_first_tier, double xl, double yl, double xr, double yr, EPosition position) 
			: ShapeFunction(is_first_tier, xl, yl, xr, yr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~InteriorShapeFunction()
		{
		}
};

#endif




