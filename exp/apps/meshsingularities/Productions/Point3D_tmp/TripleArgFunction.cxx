#include "TripleArgFunction.hxx"

using namespace D3;
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

double vertex_bot_left_near_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi2(y)*get_chi2(z);
}
double vertex_bot_left_far_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi1(y)*get_chi2(z);
}
double vertex_top_left_near_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi2(y)*get_chi1(z);
}
double vertex_top_left_far_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi1(y)*get_chi1(z);
}
double vertex_top_right_near_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi2(y)*get_chi1(z);
}
double vertex_top_right_far_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi1(y)*get_chi1(z);
}

double vertex_bot_right_near_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi2(y)*get_chi2(z);
}
double vertex_bot_right_far_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi1(y)*get_chi2(z);
}
double edge_left_near_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi2(y)*get_chi3(z);
}
double edge_left_far_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi1(y)*get_chi3(z);
}
double edge_bot_left_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi3(y)*get_chi2(z);
}
double edge_top_left_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi3(y)*get_chi1(z);
}
double edge_top_near_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi2(y)*get_chi1(z);
}
double edge_top_far_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi1(y)*get_chi1(z);
}
double edge_right_near_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi2(y)*get_chi3(z);
}
double edge_right_far_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi1(y)*get_chi3(z);
}
double edge_top_right_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi3(y)*get_chi1(z);
}
double edge_bot_right_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi3(y)*get_chi2(z);
}
double edge_bot_near_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi2(y)*get_chi2(z);
}
double edge_bot_far_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi1(y)*get_chi2(z);
}
double face_left_function(double x, double y, double z)
{
	return get_chi2(x)*get_chi3(y)*get_chi3(z);
}
double face_right_function(double x, double y, double z)
{
	return get_chi1(x)*get_chi3(y)*get_chi3(z);
}
double face_top_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi3(y)*get_chi1(z);
}
double face_bot_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi3(y)*get_chi2(z);
}
double face_near_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi2(y)*get_chi3(z);
}
double face_far_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi1(y)*get_chi3(z);
}
double interior_function(double x, double y, double z)
{
	return get_chi3(x)*get_chi3(y)*get_chi3(z);
}	


//helper functions for constraint edge nodes
double helper_edge_top_left_near_function(double x, double y, double z)
{
	return (edge_top_left_function(x,y,z) + vertex_top_left_far_function(x,y,z))/4.0;
}
double helper_edge_top_left_far_function(double x, double y, double z)
{
	return (edge_top_left_function(x,y,z) + vertex_top_left_near_function(x,y,z))/4.0;
}
double helper_edge_bot_left_near_function(double x, double y, double z)
{
	return (edge_bot_left_function(x,y,z) + vertex_bot_left_far_function(x,y,z))/4.0;
}
double helper_edge_bot_left_far_function(double x, double y, double z)
{
	return (edge_bot_left_function(x,y,z) + vertex_bot_left_near_function(x,y,z))/4.0;
}
double helper_edge_top_right_near_function(double x, double y, double z)
{
	return (edge_top_right_function(x,y,z) + vertex_top_right_far_function(x,y,z))/4.0;
}
double helper_edge_top_right_far_function(double x, double y, double z)
{
	return (edge_top_right_function(x,y,z) + vertex_top_right_near_function(x,y,z))/4.0;
}
double helper_edge_bot_right_near_function(double x, double y, double z)
{
	return (edge_bot_right_function(x,y,z) + vertex_bot_right_far_function(x,y,z))/4.0;
}
double helper_edge_bot_right_far_function(double x, double y, double z)
{
	return (edge_bot_right_function(x,y,z) + vertex_bot_right_near_function(x,y,z))/4.0;
}
double helper_edge_left_near_bot_function(double x, double y, double z)
{
	return (edge_left_near_function(x,y,z) + vertex_top_left_near_function(x,y,z))/4.0;
}
double helper_edge_left_near_top_function(double x, double y, double z)
{
	return (edge_left_near_function(x,y,z) + vertex_bot_left_near_function(x,y,z))/4.0;
}
double helper_edge_left_far_bot_function(double x, double y, double z)
{
	return (edge_left_far_function(x,y,z) + vertex_top_left_far_function(x,y,z))/4.0;
}
double helper_edge_left_far_top_function(double x, double y, double z)
{
	return (edge_left_far_function(x,y,z) + vertex_bot_left_far_function(x,y,z))/4.0;
}
double helper_edge_right_near_bot_function(double x, double y, double z)
{
	return (edge_right_near_function(x,y,z) + vertex_top_right_near_function(x,y,z))/4.0;
}
double helper_edge_right_near_top_function(double x, double y, double z)
{
	return (edge_right_near_function(x,y,z) + vertex_bot_right_near_function(x,y,z))/4.0;
}
double helper_edge_right_far_bot_function(double x, double y, double z)
{
	return (edge_right_far_function(x,y,z) + vertex_top_right_far_function(x,y,z))/4.0;
}
double helper_edge_right_far_top_function(double x, double y, double z)
{
	return (edge_right_far_function(x,y,z) + vertex_bot_right_far_function(x,y,z))/4.0;
}
double helper_edge_top_near_left_function(double x, double y, double z)
{
	return (edge_top_near_function(x,y,z) + vertex_top_right_near_function(x,y,z))/4.0;
}
double helper_edge_top_near_right_function(double x, double y, double z)
{
	return (edge_top_near_function(x,y,z) + vertex_top_left_near_function(x,y,z))/4.0;
}
double helper_edge_top_far_left_function(double x, double y, double z)
{
	return (edge_top_far_function(x,y,z) + vertex_top_right_far_function(x,y,z))/4.0;
}
double helper_edge_top_far_right_function(double x, double y, double z)
{
	return (edge_top_far_function(x,y,z) + vertex_top_left_far_function(x,y,z))/4.0;
}
double helper_edge_bot_near_left_function(double x, double y, double z)
{
	return (edge_bot_near_function(x,y,z) + vertex_bot_right_near_function(x,y,z))/4.0;
}
double helper_edge_bot_near_right_function(double x, double y, double z)
{
	return (edge_bot_near_function(x,y,z) + vertex_bot_left_near_function(x,y,z))/4.0;
}
double helper_edge_bot_far_left_function(double x, double y, double z)
{
	return (edge_bot_far_function(x,y,z) + vertex_bot_right_far_function(x,y,z))/4.0;
}
double helper_edge_bot_far_right_function(double x, double y, double z)
{
	return (edge_bot_far_function(x,y,z) + vertex_bot_left_far_function(x,y,z))/4.0;
}


double VertexBotLeftNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	z = getZValueOnElement(z);
	double value = vertex_bot_left_near_function(x,y,z);
	if(is_first_tier)
		return value; 
	switch(position)
	{
		case BOT_LEFT_NEAR:
			return value + vertex_top_left_near_function(x,y,z)/2.0 + vertex_bot_left_far_function(x,y,z)/2.0 +
					vertex_top_left_far_function(x,y,z)/4.0;
		case BOT_LEFT_FAR:
			return value / 2.0 + vertex_top_left_near_function(x,y,z)/4.0;
		case TOP_LEFT_NEAR:
			return value / 2.0 + vertex_bot_left_far_function(x,y,z)/4.0;
		case TOP_LEFT_FAR:
			return value / 4.0;
		case TOP_RIGHT_NEAR:
			return value;
		case TOP_RIGHT_FAR:
			return value;
		case BOT_RIGHT_NEAR:
			return value;
		case BOT_RIGHT_FAR:
			return value;
		return 0; 
	}
}

double VertexBotLeftFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = vertex_bot_left_far_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position)
	{
		case BOT_LEFT_NEAR:
			return value / 2.0 + vertex_top_left_far_function(x,y,z)/4.0;
		case BOT_LEFT_FAR:
			return value + vertex_bot_left_near_function(x,y,z)/2.0 + vertex_top_left_far_function(x,y,z)/2.0 +
					vertex_bot_right_far_function(x,y,z)/2.0 + vertex_top_right_far_function(x,y,z)/4.0 +
					vertex_top_left_near_function(x,y,z)/4.0;
		case TOP_LEFT_NEAR:
			return value / 4.0;
		case TOP_LEFT_FAR:
			return value / 2.0 + vertex_bot_left_near_function(x,y,z)/4.0 + vertex_bot_right_far_function(x,y,z)/4.0;
		case TOP_RIGHT_NEAR:
			return value;
		case TOP_RIGHT_FAR:
			return value/4.0;
		case BOT_RIGHT_NEAR:
			return value;
		case BOT_RIGHT_FAR:
			return value / 2.0 + vertex_top_left_far_function(x,y,z)/4.0;

	}
}

double VertexTopLeftNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = vertex_top_left_near_function(x,y,z);
	if(is_first_tier)
		return value; 
	switch(position)
	{
		case BOT_LEFT_NEAR:
			return value / 2.0 + vertex_top_left_far_function(x,y,z) / 4.0;
		case BOT_LEFT_FAR:
			return value / 4.0;
		case TOP_LEFT_NEAR:
			return value + vertex_bot_left_near_function(x,y,z)/2.0 +
					vertex_top_left_far_function(x,y,z)/2.0 + vertex_top_right_near_function(x,y,z)/2.0 +
						vertex_bot_left_far_function(x,y,z)/4.0 +vertex_top_right_far_function(x,y,z)/4.0;
		case TOP_LEFT_FAR:
			return value / 2.0 + vertex_bot_left_near_function(x,y,z)/4.0 + vertex_top_right_near_function(x,y,z)/4.0;
		case TOP_RIGHT_NEAR:
			return value / 2.0 + vertex_top_left_far_function(x,y,z)/4.0;
		case TOP_RIGHT_FAR:
			return value/4.0;
		case BOT_RIGHT_FAR:
			return value;
		case BOT_RIGHT_NEAR:
			return value; 
	}
}

double VertexTopLeftFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = vertex_top_left_far_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position)
	{
		case BOT_LEFT_NEAR:
			return value / 4.0;
		case BOT_LEFT_FAR:
			return value / 2.0 + vertex_top_right_far_function(x,y,z)/4.0 + vertex_top_left_near_function(x,y,z)/4.0;
		case TOP_LEFT_NEAR:
			return value / 2.0 + vertex_bot_left_far_function(x,y,z)/4.0 + vertex_top_right_far_function(x,y,z)/4.0;
		case TOP_LEFT_FAR:
			return value + vertex_top_left_near_function(x,y,z)/2.0 + vertex_top_right_far_function(x,y,z)/2.0
					+ vertex_bot_left_far_function(x,y,z)/2.0 + vertex_bot_left_near_function(x,y,z)/4.0 +
					vertex_bot_right_far_function(x,y,z)/4.0 + vertex_top_right_near_function(x,y,z)/4.0;

		case TOP_RIGHT_NEAR:
			return value / 4.0;
		case TOP_RIGHT_FAR:
			return value / 2.0 + vertex_top_left_near_function(x,y,z)/4.0 + vertex_bot_left_far_function(x,y,z)/4.0;
		case BOT_RIGHT_FAR:
			return value / 4.0;
		case BOT_RIGHT_NEAR:
			return value;
	}
}

double VertexTopRightNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value =  vertex_top_right_near_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position)
	{
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return value;
	case TOP_LEFT_NEAR:
		return value/2.0 + vertex_top_right_far_function(x,y,z)/4.0;
	case TOP_LEFT_FAR:
		return value/4.0;
	case TOP_RIGHT_NEAR:
		return value  + vertex_top_left_near_function(x,y,z)/2.0 + vertex_top_right_far_function(x,y,z)/2.0 +
				vertex_top_left_far_function(x,y,z)/4.0;
	case TOP_RIGHT_FAR:
		return value/2.0 + vertex_top_left_near_function(x,y,z)/4.0;
	case BOT_RIGHT_FAR:
		return value;
	case BOT_RIGHT_NEAR:
		return value;
	}
}


double VertexTopRightFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	z = getZValueOnElement(z);
	double value =  vertex_top_right_far_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position)
	{
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return value/4.0;
	case TOP_LEFT_NEAR:
		return value/4.0;
	case TOP_LEFT_FAR:
		return value/2.0 + vertex_top_right_near_function(x,y,z)/4.0 + vertex_bot_right_far_function(x,y,z)/4.0;
	case TOP_RIGHT_NEAR:
		return value / 2.0 + vertex_top_left_far_function(x,y,z)/4.0;
	case TOP_RIGHT_FAR:
		return value + vertex_top_left_far_function(x,y,z)/2.0 + vertex_bot_right_far_function(x,y,z)/2.0
				+ vertex_top_right_near_function(x,y,z)/2.0 + vertex_top_left_near_function(x,y,z)/4.0 +
				vertex_bot_left_far_function(x,y,z)/4.0;
	case BOT_RIGHT_FAR:
		return value / 2.0 + vertex_top_left_far_function(x,y,z)/4.0;
	case BOT_RIGHT_NEAR:
		return value;
	}
}

double VertexBotRightFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value =  vertex_bot_right_far_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position)
	{
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return value/2.0 + vertex_top_right_far_function(x,y,z)/4.0;
	case TOP_LEFT_NEAR:
		return value;
	case TOP_LEFT_FAR:
		return value/4.0;
	case TOP_RIGHT_NEAR:
		return value;
	case TOP_RIGHT_FAR:
		return value/2.0 + vertex_bot_left_far_function(x,y,z)/4.0;
	case BOT_RIGHT_FAR:
		return value + vertex_bot_left_far_function(x,y,z)/2.0 + vertex_top_right_far_function(x,y,z)/2.0 +
				vertex_top_left_far_function(x,y,z)/4.0;
	case BOT_RIGHT_NEAR:
		return value;
	}
}

double VertexBotRightNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	z = getZValueOnElement(z);
	return vertex_bot_right_near_function(x,y,z);
	
}


double EdgeBotLeftShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	z = getZValueOnElement(z);
	double value = edge_bot_left_function(x,y,z);
	if(is_first_tier)
			return value;
	switch(position)
	{
	case BOT_LEFT_NEAR:
		return helper_edge_bot_left_near_function(x,y,z) + helper_edge_top_left_near_function(x,y,z)/2.0;
	case BOT_LEFT_FAR:
		return helper_edge_bot_left_far_function(x,y,z) + helper_edge_top_left_far_function(x,y,z)/2.0;
	case TOP_LEFT_NEAR:
		return helper_edge_bot_left_near_function(x,y,z)/2.0;
	case TOP_LEFT_FAR:
			return helper_edge_bot_left_far_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return value;
	case TOP_RIGHT_FAR:
		return value;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return value;
	}

}

double EdgeTopLeftShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_top_left_function(x,y,z);
	if(is_first_tier)
			return value;
	switch(position)
	{
	case BOT_LEFT_NEAR:
		return helper_edge_top_left_near_function(x,y,z)/2.0;
	case BOT_LEFT_FAR:
		return helper_edge_top_left_far_function(x,y,z)/2.0;
	case TOP_LEFT_NEAR:
		return helper_edge_top_left_near_function(x,y,z) +
				helper_edge_bot_left_near_function(x,y,z)/2.0 + helper_edge_top_right_near_function(x,y,z)/2.0;
	case TOP_LEFT_FAR:
		return helper_edge_top_left_far_function(x,y,z) +
				helper_edge_bot_left_far_function(x,y,z)/2.0 + helper_edge_top_right_far_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return helper_edge_top_left_near_function(x,y,z)/2.0;
	case TOP_RIGHT_FAR:
		return helper_edge_top_left_far_function(x,y,z)/2.0;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return value;
	}
	
}

double EdgeTopRightShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_top_right_function(x,y,z);
	if(is_first_tier)
			return value;
	switch(position)
	{
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return value;
	case TOP_LEFT_NEAR:
		return helper_edge_top_right_near_function(x,y,z)/2.0;
	case TOP_LEFT_FAR:
		return helper_edge_top_right_far_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return helper_edge_top_right_near_function(x,y,z) + helper_edge_top_left_near_function(x,y,z)/2.0;
	case TOP_RIGHT_FAR:
		return helper_edge_top_right_far_function(x,y,z) + helper_edge_top_left_far_function(x,y,z)/2.0;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return value;
	}

}

double EdgeBotRightShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_bot_right_function(x,y,z);
	return value;

}


double EdgeTopNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y); 
	z = getZValueOnElement(z);
	double value = edge_top_near_function(x,y,z);
	if(is_first_tier)
			return value;
	switch(position){
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return value;
	case TOP_LEFT_NEAR:
		return helper_edge_top_near_left_function(x,y,z) + helper_edge_top_far_left_function(x,y,z)/2.0;
	case TOP_LEFT_FAR:
		return helper_edge_top_near_left_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return helper_edge_top_near_right_function(x,y,z) + helper_edge_top_far_right_function(x,y,z)/2.0;
	case TOP_RIGHT_FAR:
		return helper_edge_top_near_right_function(x,y,z)/2.0;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return value;
	}

}

double EdgeBotNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_bot_near_function(x,y,z);
	return value;

}

double EdgeTopFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_top_far_function(x,y,z);
	if(is_first_tier)
			return value;
	switch(position){
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return helper_edge_top_far_left_function(x,y,z)/2.0;
	case TOP_LEFT_NEAR:
		return helper_edge_top_far_left_function(x,y,z)/2.0;
	case TOP_LEFT_FAR:
		return helper_edge_top_far_left_function(x,y,z) +
				helper_edge_bot_far_left_function(x,y,z)/2.0 + helper_edge_top_near_left_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return helper_edge_top_far_right_function(x,y,z)/2.0;
	case TOP_RIGHT_FAR:
		return helper_edge_top_far_right_function(x,y,z) +
					helper_edge_bot_far_right_function(x,y,z)/2.0 + helper_edge_top_near_right_function(x,y,z)/2.0;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return helper_edge_top_far_right_function(x,y,z)/2.0;
	}
	
}

double EdgeBotFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_bot_far_function(x,y,z);
	if(is_first_tier)
			return value;
	switch(position){
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return helper_edge_bot_far_left_function(x,y,z) + helper_edge_top_far_left_function(x,y,z)/2.0;
	case TOP_LEFT_NEAR:
		return value;
	case TOP_LEFT_FAR:
		return helper_edge_bot_far_left_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return value;
	case TOP_RIGHT_FAR:
		return helper_edge_bot_far_right_function(x,y,z)/2.0;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return helper_edge_bot_far_right_function(x,y,z) + helper_edge_top_far_right_function(x,y,z)/2.0;
	}

}

double EdgeLeftNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_left_near_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position){
	case BOT_LEFT_NEAR:
		return helper_edge_left_near_bot_function(x,y,z) + helper_edge_left_far_bot_function(x,y,z)/2.0;
	case BOT_LEFT_FAR:
		return helper_edge_left_near_bot_function(x,y,z)/2.0;
	case TOP_LEFT_NEAR:
		return helper_edge_left_near_top_function(x,y,z) + helper_edge_left_far_top_function(x,y,z)/2.0;
	case TOP_LEFT_FAR:
		return helper_edge_left_near_top_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return value;
	case TOP_RIGHT_FAR:
		return value;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return value;
	}

}

double EdgeLeftFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_left_far_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position){
	case BOT_LEFT_NEAR:
		return helper_edge_left_far_bot_function(x,y,z)/2.0;
	case BOT_LEFT_FAR:
		return helper_edge_left_far_bot_function(x,y,z) +
				helper_edge_left_near_bot_function(x,y,z)/2.0 + helper_edge_right_far_bot_function(x,y,z)/2.0;
	case TOP_LEFT_NEAR:
		return helper_edge_left_far_top_function(x,y,z)/2.0;
	case TOP_LEFT_FAR:
		return helper_edge_left_far_top_function(x,y,z) +
				helper_edge_left_near_top_function(x,y,z)/2.0 + helper_edge_right_far_top_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return value;
	case TOP_RIGHT_FAR:
		return helper_edge_left_far_top_function(x,y,z)/2.0;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return helper_edge_left_far_bot_function(x,y,z)/2.0;
	}

}

double EdgeRightFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_right_far_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position){
	case BOT_LEFT_NEAR:
		return value;
	case BOT_LEFT_FAR:
		return helper_edge_right_far_bot_function(x,y,z)/2.0;
	case TOP_LEFT_NEAR:
		return value;
	case TOP_LEFT_FAR:
		return helper_edge_right_far_top_function(x,y,z)/2.0;
	case TOP_RIGHT_NEAR:
		return value;
	case TOP_RIGHT_FAR:
		return helper_edge_right_far_top_function(x,y,z) + helper_edge_left_far_top_function(x,y,z)/2.0;
	case BOT_RIGHT_NEAR:
		return value;
	case BOT_RIGHT_FAR:
		return helper_edge_right_far_bot_function(x,y,z) + helper_edge_left_far_bot_function(x,y,z)/2.0;
	}

}


double EdgeRightNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = edge_right_near_function(x,y,z);
	return value;

}

double FaceLeftShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = face_left_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position){
	case BOT_LEFT_NEAR:
		return (vertex_top_left_far_function(x,y,z) + edge_top_left_function(x,y,z) + edge_left_far_function(x,y,z)
				+ face_left_function(x,y,z))/16.0;
	case BOT_LEFT_FAR:
		return (vertex_top_left_near_function(x,y,z) + edge_top_left_function(x,y,z) + edge_left_near_function(x,y,z)
					+ face_left_function(x,y,z))/16.0;
	case TOP_LEFT_NEAR:
		return (vertex_bot_left_far_function(x,y,z) + edge_bot_left_function(x,y,z) + edge_left_far_function(x,y,z)
					+ face_left_function(x,y,z))/16.0;
	case TOP_LEFT_FAR:
		return (vertex_bot_left_near_function(x,y,z) + edge_bot_left_function(x,y,z) + edge_left_near_function(x,y,z)
				+ face_left_function(x,y,z))/16.0;
	default:
		return value;
	}

}

double FaceRightShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = face_right_function(x,y,z);
	return value;

}

double FaceTopShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = face_top_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position){
	case TOP_LEFT_NEAR:
		return (vertex_top_right_far_function(x,y,z) + edge_top_far_function(x,y,z) + edge_top_right_function(x,y,z)
			+ face_top_function(x,y,z))/16.0;
	case TOP_LEFT_FAR:
		return (vertex_top_right_near_function(x,y,z) + edge_top_near_function(x,y,z) + edge_top_right_function(x,y,z)
				+ face_top_function(x,y,z))/16.0;
	case TOP_RIGHT_NEAR:
		return (vertex_top_left_far_function(x,y,z) + edge_top_far_function(x,y,z) + edge_top_left_function(x,y,z)
				+ face_top_function(x,y,z))/16.0;
	case TOP_RIGHT_FAR:
		return (vertex_top_left_near_function(x,y,z) + edge_top_near_function(x,y,z) + edge_top_left_function(x,y,z)
				+ face_top_function(x,y,z))/16.0;
	default:
		return value;
	}
}

double FaceBotShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = face_bot_function(x,y,z);
	return value;

}

double FaceFarShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = face_far_function(x,y,z);
	if(is_first_tier)
		return value;
	switch(position){
	case BOT_LEFT_FAR:
		return (vertex_top_right_far_function(x,y,z) + edge_top_far_function(x,y,z) + edge_right_far_function(x,y,z) +
				face_far_function(x,y,z))/16.0;
	case TOP_LEFT_FAR:
		return (vertex_bot_right_far_function(x,y,z) + edge_bot_far_function(x,y,z) + edge_right_far_function(x,y,z) +
					face_far_function(x,y,z))/16.0;
	case TOP_RIGHT_FAR:
		return (vertex_bot_left_far_function(x,y,z) + edge_bot_far_function(x,y,z) + edge_left_far_function(x,y,z) +
					face_far_function(x,y,z))/16.0;
	case BOT_RIGHT_FAR:
		return (vertex_top_left_far_function(x,y,z) + edge_top_far_function(x,y,z) + edge_left_far_function(x,y,z) +
					face_far_function(x,y,z))/16.0;
	default:
		return value;
	}

}

double FaceNearShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x);
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	double value = face_near_function(x,y,z);
	return value;

}


double InteriorShapeFunction::ComputeValue(double x, double y, double z)
{
	x = getXValueOnElement(x); 
	y = getYValueOnElement(y);
	z = getZValueOnElement(z);
	return interior_function(x,y,z);
}






