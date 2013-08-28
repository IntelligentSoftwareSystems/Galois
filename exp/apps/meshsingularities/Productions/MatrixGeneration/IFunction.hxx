

class IFunction
{
	protected:
		double* coordinates;
		bool* neighbours;
		bool is_first_tier;


	public:
		IFunction(double* coordinates, bool is_first_tier, bool* neighbours) :
			coordinates(coordinates), is_first_tier(is_first_tier), neighbours(neighbours)
		{

		}
		IFunction()
		{

		}

		virtual ~IFunction()
		{

		}
};
