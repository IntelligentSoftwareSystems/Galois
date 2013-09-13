

class Function
{
	public:
		double* coordinates;
		bool* neighbours;



	public:
		Function(double* coordinates, bool is_first_tier, bool* neighbours) :
			coordinates(coordinates), neighbours(neighbours)
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
