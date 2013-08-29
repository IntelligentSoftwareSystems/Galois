#ifndef TASKDESCRIPTION_H
#define TASKDESCRIPTION_H

struct TaskDescription {
	int dimensions;
	int nrOfTiers;
	double size;
	bool quad;
	double x;
	double y;
	double z;

	double (*function)(int, ...);

	bool performTests;
};

#endif
