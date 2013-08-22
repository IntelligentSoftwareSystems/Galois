#ifndef __TRIPLEARGFUNCTION_H_INCLUDED
#define __TRIPLEARGFUNCTION_H_INCLUDED
#include "EPosition.hxx"
#include <stdarg.h>
#include <vector>
#include <stdio.h>
namespace D3{

double get_chi1(double var);
double get_chi2(double var);
double get_chi3(double var);

class NArgFunction
{
	public:
		virtual double ComputeValue(const std::vector<double>& p) = 0;

		virtual ~NArgFunction()
		{

		}
};


class ITripleArgFunction //: public NArgFunction
{
	
	public:
		virtual double ComputeValue(double x, double y, double z) = 0;


		virtual double ComputeValue(const std::vector<double>& p)
		{
			std::vector<double>::const_iterator it = p.begin();
			return ComputeValue(*(it),*(it+1),*(it+2));
		}

		virtual ~ITripleArgFunction()
		{

		}
};

class TripleArgFunctionWrapper : public ITripleArgFunction
{
	private:
		double (*f)(int,...);
	public:
		virtual double ComputeValue(double x, double y, double z)
		{
			return (*f)(3,x,y,z);
		}

		TripleArgFunctionWrapper(double (*f)(int,...)) : f(f)
		{

		}

		virtual ~TripleArgFunctionWrapper()
		{

		}
};


class ShapeFunction : public ITripleArgFunction
{
	protected :
		double xl; 
		double yl; 
		double xr; 
		double yr;
		double zl;
		double zr;
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
		
		double getZValueOnElement(double z)
		{
			return (z-zl)/(zr-zl);
		}
	
	public:
		ShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			:  xl(xl), yl(yl), xr(xr), yr(yr), zl(zl), zr(zr), is_first_tier(is_first_tier), position(position)
		{
			
		}
		

};


class VertexBotLeftNearShapeFunction : public ShapeFunction
{
	public:

		VertexBotLeftNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);

		~VertexBotLeftNearShapeFunction()
		{
		}

};

class VertexBotLeftFarShapeFunction : public ShapeFunction
{
	public:

		VertexBotLeftFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		
		~VertexBotLeftFarShapeFunction()
		{
		}
	
};

class VertexTopLeftNearShapeFunction : public ShapeFunction
{
	public:

		VertexTopLeftNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~VertexTopLeftNearShapeFunction()
		{
		}
};

class VertexTopLeftFarShapeFunction : public ShapeFunction
{
	public:

		VertexTopLeftFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~VertexTopLeftFarShapeFunction()
		{
		}
};

class VertexTopRightNearShapeFunction : public ShapeFunction
{
	public:

		VertexTopRightNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		virtual ~VertexTopRightNearShapeFunction()
		{
		}
};

class VertexTopRightFarShapeFunction : public ShapeFunction
{
	public:

		VertexTopRightFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		virtual ~VertexTopRightFarShapeFunction()
		{
		}
};



class VertexBotRightNearShapeFunction : public ShapeFunction
{
	public:

		VertexBotRightNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		~VertexBotRightNearShapeFunction()
		{
		}
};

class VertexBotRightFarShapeFunction : public ShapeFunction
{
	public:

		VertexBotRightFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		~VertexBotRightFarShapeFunction()
		{
		}
};


class EdgeBotLeftShapeFunction : public ShapeFunction
{
	public:

		EdgeBotLeftShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
				: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
			{

			}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeBotLeftShapeFunction()
		{
		}
};

class EdgeTopLeftShapeFunction : public ShapeFunction
{
	public:

		EdgeTopLeftShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
				: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
			{

			}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeTopLeftShapeFunction()
		{
		}
};


class EdgeLeftNearShapeFunction : public ShapeFunction
{
	public:

		EdgeLeftNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
				: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
			{

			}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeLeftNearShapeFunction()
		{
		}
};

class EdgeLeftFarShapeFunction : public ShapeFunction
{
	public:
	
		EdgeLeftFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
				: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
			{
				
			}
			
		virtual double ComputeValue(double x, double y, double z);
		~EdgeLeftFarShapeFunction()
		{
		}
};



class EdgeTopNearShapeFunction : public ShapeFunction
{
	public:

		EdgeTopNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeTopNearShapeFunction()
		{
		}
};

class EdgeTopFarShapeFunction : public ShapeFunction
{
	public:

		EdgeTopFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeTopFarShapeFunction()
		{
		}
};


class EdgeBotNearShapeFunction : public ShapeFunction
{
	public:

		EdgeBotNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		~EdgeBotNearShapeFunction()
		{
		}
};

class EdgeBotFarShapeFunction : public ShapeFunction
{
	public:

		EdgeBotFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		~EdgeBotFarShapeFunction()
		{
		}
};

class EdgeBotRightShapeFunction : public ShapeFunction
{
	public:

		EdgeBotRightShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeBotRightShapeFunction()
		{
		}
};

class EdgeTopRightShapeFunction : public ShapeFunction
{
	public:

		EdgeTopRightShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeTopRightShapeFunction()
		{
		}
};




class EdgeRightNearShapeFunction : public ShapeFunction
{
	public:

		EdgeRightNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~EdgeRightNearShapeFunction()
		{
		}
};

class EdgeRightFarShapeFunction : public ShapeFunction
{	
	public:

		EdgeRightFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		~EdgeRightFarShapeFunction()
		{
		}
};

class FaceLeftShapeFunction : public ShapeFunction
{
	public:

		FaceLeftShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~FaceLeftShapeFunction()
		{
		}
};

class FaceRightShapeFunction : public ShapeFunction
{
	public:

		FaceRightShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~FaceRightShapeFunction()
		{
		}
};

class FaceTopShapeFunction : public ShapeFunction
{
	public:

		FaceTopShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~FaceTopShapeFunction()
		{
		}
};

class FaceBotShapeFunction : public ShapeFunction
{
	public:

		FaceBotShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~FaceBotShapeFunction()
		{
		}
};

class FaceFarShapeFunction : public ShapeFunction
{
	public:

		FaceFarShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~FaceFarShapeFunction()
		{
		}
};

class FaceNearShapeFunction : public ShapeFunction
{
	public:

		FaceNearShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{

		}

		virtual double ComputeValue(double x, double y, double z);
		~FaceNearShapeFunction()
		{
		}
};


class InteriorShapeFunction : public ShapeFunction
{
	public:

		InteriorShapeFunction(bool is_first_tier, double xl, double yl, double zl, double xr, double yr, double zr, EPosition position)
			: ShapeFunction(is_first_tier, xl, yl, zl, xr, yr, zr, position)
		{
			
		}
		
		virtual double ComputeValue(double x, double y, double z);
		~InteriorShapeFunction()
		{
		}
};
}
#endif




