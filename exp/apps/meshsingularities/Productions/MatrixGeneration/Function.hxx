

class Function
{
	public:
		double* coordinates;
		bool* neighbours;
		bool is_first_tier;


	public:
		Function(double* coordinates, bool is_first_tier, bool* neighbours) :
			coordinates(coordinates), is_first_tier(is_first_tier), neighbours(neighbours)
		{

		}
		Function()
		{

		}

		virtual ~Function()
		{
			//delete[] coordinates;
			//delete[] neighbours;
		}
};
