#include "DoubleArgFunction.hxx"
using namespace D2;

namespace D2{
	double get_chi1(double var)
	{
		return var;
	}
	double get_chi2(double var)
	{
		return 1-var;
	}
	double get_chi3(double var)
	{
		return var*(1-var);
	}
}

double vertex_bot_left_function(double x, double y)
{
	return get_chi2(x)*get_chi2(y);
}
double vertex_top_left_function(double x, double y)
{
	return get_chi2(x)*get_chi1(y);
}
double vertex_top_right_function(double x, double y)
{
	return get_chi1(x)*get_chi1(y);
}
double vertex_bot_right_function(double x, double y)
{
	return get_chi1(x)*get_chi2(y);
}
double edge_left_function(double x, double y)
{
	return get_chi2(x)*get_chi3(y); 
}
double edge_top_function(double x, double y)
{
	return get_chi3(x)*get_chi1(y); 
}
double edge_right_function(double x, double y)
{
	return get_chi1(x)*get_chi3(y); 
}
double edge_bot_function(double x, double y)
{
	return get_chi3(x)*get_chi2(y); 
}
double interior_function(double x, double y)
{
	return get_chi3(x)*get_chi3(y); 
}	


double VertexBotLeftShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value = vertex_bot_left_function(x,y);

	switch(position)
	{
		case BOT_LEFT:
			if(!neighbours[LEFT])
				value += vertex_top_left_function(x,y)/2.0;
			if(!neighbours[BOT])
				value += vertex_bot_right_function(x,y)/2.0;
			break;
		case TOP_LEFT:
			if(!neighbours[LEFT])
				return value/2.0;
			break;
		case TOP_RIGHT:
			break;
		case BOT_RIGHT:
			if(!neighbours[BOT])
				return value/2.0;
			break;
	}
	return value;
}

double VertexTopLeftShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value = vertex_top_left_function(x,y);

	switch(position)
	{
		case BOT_LEFT:
			if(!neighbours[LEFT])
				return value/2.0;
			break;
		case TOP_LEFT:
			if(!neighbours[LEFT])
				value += vertex_bot_left_function(x,y)/2.0;
			if(!neighbours[TOP])
				value += vertex_top_right_function(x,y)/2.0;
			break;
		case TOP_RIGHT:
			if(!neighbours[TOP])
				return value/2.0;
			break;
		case BOT_RIGHT:
			return value;
	}
	return value;
}


double VertexTopRightShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value =  vertex_top_right_function(x,y);

	switch(position)
	{
	case BOT_LEFT:
		return value;
	case TOP_LEFT:
		if(!neighbours[TOP])
			return value/2.0;
		break;
	case TOP_RIGHT:
		if(!neighbours[TOP])
			value+= vertex_top_left_function(x,y)/2.0;
		if(!neighbours[RIGHT])
			value+= vertex_bot_right_function(x,y)/2.0;
		break;
	case BOT_RIGHT:
		if(!neighbours[RIGHT])
			return value/2.0;
		break;
	}

	return value;
}


double VertexBotRightShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value = vertex_bot_right_function(x,y);

	switch(position)
	{
	case BOT_LEFT:
		if(!neighbours[BOT])
			return value/2.0;
		break;
	case TOP_LEFT:
		return value;
	case TOP_RIGHT:
		if(!neighbours[RIGHT])
			return value/2.0;
		break;
	case BOT_RIGHT:
		if(!neighbours[BOT])
			value += vertex_bot_left_function(x,y)/2.0;
		if(!neighbours[RIGHT])
			value += vertex_top_right_function(x,y)/2.0;
		break;
	
	}

	return value;
}


double EdgeLeftShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value = edge_left_function(x,y); 

	switch(position)
	{
	case BOT_LEFT: 
		if(!neighbours[LEFT])
			return (value + vertex_top_left_function(x,y))*0.25;
		return value;
	case TOP_LEFT:
		if(!neighbours[LEFT])
			return (value + vertex_bot_left_function(x,y))*0.25;
		return value;
	case TOP_RIGHT:
		return value; 
	case BOT_RIGHT:
		return value;
	}
	return value;
}

double EdgeTopShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value = edge_top_function(x,y); 

	switch(position){
	case BOT_LEFT:
		return value;
	case TOP_LEFT:
		if(!neighbours[TOP])
			return (value + vertex_top_right_function(x,y))*0.25;
		return value;
	case TOP_RIGHT:
		if(!neighbours[TOP])
			return (value + vertex_top_left_function(x,y))*0.25;
		return value;
	case BOT_RIGHT:
		return value;
	}
	return value;
}


double EdgeBotShapeFunction::ComputeValue(double x, double y)
{

	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value = edge_bot_function(x,y);

	switch(position){
	case BOT_LEFT:
		if(!neighbours[BOT])
			return (value + vertex_bot_right_function(x,y))*0.25;
		return value;
	case TOP_LEFT:
		return value;
	case TOP_RIGHT:
		return value;
	case BOT_RIGHT:
		if(!neighbours[BOT])
			return (value + vertex_bot_left_function(x,y))*0.25;
		return value;
	}
	return value;
}

double EdgeRightShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	double value = edge_right_function(x,y);

	switch(position){
	case BOT_LEFT:
		return value;
	case TOP_LEFT:
		return value;
	case TOP_RIGHT:
		if(!neighbours[RIGHT])
			return (value + vertex_bot_right_function(x,y))*0.25;
		return value;
	case BOT_RIGHT:
		if(!neighbours[RIGHT])
			return (value + vertex_top_right_function(x,y))*0.25;
		return value;
	}
	return value;
}

double InteriorShapeFunction::ComputeValue(double x, double y)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	return interior_function(x,y);
}






