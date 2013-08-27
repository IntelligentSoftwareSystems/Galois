#ifndef __DOUBLEARGFUNCTION_H_INCLUDED
#define __DOUBLEARGFUNCTION_H_INCLUDED
#include "EPosition.hxx"
#include "NPosition.hxx"
#include "../MatrixGeneration/Function.hxx"
#include <stdio.h>

namespace D2
{

double get_chi1(double var);
double get_chi2(double var);
double get_chi3(double var);


class IDoubleArgFunction : public Function
{
	
	public:
		virtual double ComputeValue(double x, double y) = 0; 

	IDoubleArgFunction(double* coordinates, bool is_first_tier, bool* neighbours) : Function(coordinates,is_first_tier,neighbours)
	{

	}
	IDoubleArgFunction()
	{

	}
	virtual ~IDoubleArgFunction()
	{

	}
};

class DoubleArgFunctionWrapper : public IDoubleArgFunction
{
	private:
		double (*f)(int,...);
	public:
		virtual double ComputeValue(double x, double y)
		{
			return (*f)(2,x,y);
		}

		DoubleArgFunctionWrapper(double (*f)(int,...)) : f(f)
		{

		}

		virtual ~DoubleArgFunctionWrapper()
		{

		}

};


class ShapeFunction : public IDoubleArgFunction
{
	protected :
		double xl; 
		double yl; 
		double xr; 
		double yr;
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
		ShapeFunction(double* coordinates, bool is_first_tier, bool* neighbours, EPosition position)
			:  IDoubleArgFunction(coordinates, is_first_tier, neighbours), position(position)
		{
			xl = coordinates[0];
			xr = coordinates[1];
			yl = coordinates[2];
			yr = coordinates[3];
		}
		

};


class VertexBotLeftShapeFunction : public ShapeFunction
{
	public:

		VertexBotLeftShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
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

		VertexTopLeftShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
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

		VertexTopRightShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		virtual ~VertexTopRightShapeFunction()
		{
		}
};

class VertexBotRightShapeFunction : public ShapeFunction
{
	public:

		VertexBotRightShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
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
	
		EdgeLeftShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
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

		EdgeTopShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
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

		EdgeBotShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
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

		EdgeRightShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
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

		InteriorShapeFunction(bool is_first_tier, double* coordinates, bool* neighbours, EPosition position)
			: ShapeFunction(coordinates, is_first_tier, neighbours, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y);
		~InteriorShapeFunction()
		{
		}
};
}
#endif




