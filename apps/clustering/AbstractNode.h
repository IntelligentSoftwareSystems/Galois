/*
 * AbstractNode.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include <stdlib.h>
#include"RandomGenerator.h"
#ifndef ABSTRACTNODE_H_
#define ABSTRACTNODE_H_
typedef std::vector<float> TimeVector;

class AbstractNode {

public:
	int nodeIdAndFlags;
	/**
	 * position of point or center of cluster's bounding box (or to-vector for infinite lights)
	 */
	float x;
	float y;
	float z;
	short startTime;
	short endTime;
	/**
	 * Number of representatives stored per node
	 */
	static int globalNumReps;// = -1;
	/**
	 * Does each representative represent a different time instant or are all at the same time instant
	 */
	static bool globalMultitime;// = false;
	static std::vector<TimeVector> repRandomNums;
	/**
	 * Intensity or strength of this point or cluster
	 */
private:
	float intensityRed, intensityGreen, intensityBlue; //total intensity
	//start and end times represented in timeVector vector
	TimeVector timeVector; //fractional intensity per time interval
	//if startTime==endTime then the Intensity time is always the same and points to this one
	static const TimeVector singleTimeVector;// = { 1.0f };
	static const int ML_CLAMP;// = 0x02; //subject to clamping?
	static const int ML_COMBO_MASK;// = 0xFC; //flags that should be shared with a parent
	static const int ML_MATCH_MASK;// = 0xFC;
	static const int ML_ID_SHIFT;//T = 8;
public:
	/**
	 * Creates a new instance of MLIntensity
	 */
	AbstractNode() {
		startTime = -1; //deliberately invalid value
		nodeIdAndFlags = -1 << AbstractNode::ML_ID_SHIFT; //must be set to a correct value later
	}

	/**
	 * Get the total intensity of this light or cluster
	 * as a scalar (ie averaged down to a single number)
	 *
	 * @return Scalar total intensity
	 */
	float getScalarTotalIntensity() {
		return (1.0f / 3.0f) * (intensityRed + intensityGreen + intensityBlue);
	}

	/**
	 * What fraction of the total intensity was at the given time instant
	 */
//	float getRelativeIntensity(int inTime) {
//		if (inTime < startTime || inTime > endTime) {
//			return 0;
//		}
//		return timeVector[inTime - startTime];
//	}
	/**
	 * Set the Intensity of this node to be equal to the specified
	 * spectrum scaled by the specified factor and at the single instant specified
	 *
	 * @param inScaleFactor Factor to scale spectrum by before setting Intensity
	 */
	void setIntensity(double inScaleFactor, short inTime) {
		intensityRed = (float) inScaleFactor;
		intensityGreen = (float) inScaleFactor;
		intensityBlue = (float) inScaleFactor;
		if (inTime == -1) {
			inTime = 0;
		}
		if (inTime >= 0) {
			startTime = inTime;
			endTime = inTime;
			timeVector = singleTimeVector;
		} else {
			//negative value used as signal that should be uniform across all time
			int len = -inTime;
			startTime = 0;
			endTime = (short) (len - 1);
			timeVector.clear();
			timeVector.resize(len);//= new float[len];
			for (unsigned int i = 0; i < timeVector.size(); i++)
				timeVector[i] = (1.0f) / len;
			//Arrays.fill(timeVector, 1.0f / len);
			scaleIntensity(len);
		}
	}

	/**
	 * Set this nodes maximum intensity to be the sum of the maximum intensities
	 * of the two specified nodes.
	 *
	 * @param inNodeA Node to sum maximum intensity from
	 * @param inNodeB Node to sum maximum intensity from
	 */
	void setSummedIntensity(AbstractNode & inA, AbstractNode & inB) {
		intensityRed = inA.intensityRed + inB.intensityRed;
		intensityGreen = inA.intensityGreen + inB.intensityGreen;
		intensityBlue = inA.intensityBlue + inB.intensityBlue;
		startTime = inA.startTime < inB.startTime ? inA.startTime : inB.endTime;
		endTime = inA.startTime < inB.startTime ? inB.startTime : inA.endTime;
		if (startTime != endTime) {
			unsigned int len = endTime - startTime + 1;
			if (timeVector.size() < len) {
				timeVector.clear();
				timeVector.resize(len);//= new float[len];
			} else {
				for (unsigned int i = 0; i < timeVector.size(); i++) {
					timeVector[i] = 0;
				}
			}
			float weightA = inA.getScalarTotalIntensity();
			float weightB = inB.getScalarTotalIntensity();
			float invDenom = 1.0f / (weightA + weightB);
			weightA *= invDenom;
			weightB *= invDenom;
			for (int i = inA.startTime; i <= inA.endTime; i++) {
				timeVector[i - startTime] += weightA * inA.timeVector[i
						- inA.startTime];
			}
			for (int i = inB.startTime; i <= inB.endTime; i++) {
				timeVector[i - startTime] += weightB * inB.timeVector[i
						- inB.startTime];
			}
		} else {
			timeVector = AbstractNode::singleTimeVector;
		}
	}

	/**
	 * Scale the maximum intensity of this node by a given factor
	 *
	 * @param inScale Factor to scale maximum intensity by
	 */
	void scaleIntensity(double inScale) {
		float scale = (float) inScale;
		intensityRed *= scale;
		intensityGreen *= scale;
		intensityBlue *= scale;
	}

	static void setGlobalNumReps() {
		if (globalNumReps == 1) {
			//nothing changed
			return;
		}
		//trees must be rebuilt for this to take effect
		globalNumReps = 1;
		double inc = 1.0f / 1;
		RandomGenerator ranGen(452389425623145845L);
		repRandomNums.clear(); //= new float[256][];
		repRandomNums.resize(256);
		for (int i = 0; i < 256; i++) {
			std::vector<float> * ranVec = new std::vector<float>(1);// = new float[1];
//			std::cout<<"Starting "<<i<<std::endl;
			//fill vector with uniform randomized numbers (uniformly distributed, jittered)
			//					for (unsigned int j = 0; j < ranVec.size(); j++)
			//					{
			//						ranVec[j] = (float) ((j + ranGen.nextDouble()) * inc);
			//					}
			(*ranVec)[0] = (float) ((0 + ranGen.nextDouble()) * inc);
			//now randomly permute the numbers
			//					for (unsigned int j = ranVec.size() - 1; j > 0; j--)
			//					{
			//						unsigned int index = (unsigned int) ((j + 1) * ranGen.nextDouble());
			//						if (index > j) {
			//							//throw new RuntimeException("badness " + index);
			//							std::cout << "Error in SetGlobalReps" << std::endl;
			//						}
			//						//swap index element with jth element
			//						float temp = ranVec[j];
			//						ranVec[j] = ranVec[index];
			//						ranVec[index] = temp;
			//					}
			int index = (int) ((0 + 1) * ranGen.nextDouble());
			if (index > 0) {
				//throw new RuntimeException("badness " + index);
				std::cout << "Error in SetGlobalReps" << std::endl;
			}
			//swap index element with jth element
//			std::cout<<"Done "<<i<<" :: "<<(*ranVec)[0]<< " inc "<<inc<<"  "<<ranGen.nextDouble()<<std::endl;
			float temp = (*ranVec)[0];
			(*ranVec)[0] = (*ranVec)[index];
			(*ranVec)[index] = temp;
			//that's all now store the random vector for later use
			repRandomNums[i] = *ranVec;
		}
		/*
		for(int i=0;i<256;i++){
			std::cout<<" V:: "<< (repRandomNums[i])[0] << std::endl;
		}
		*/
	}

	static void setGlobalMultitime() {
		//trees must be rebuilt for this to take effect
		globalMultitime = false;
	}

	float getX() {
		return x;
	}

	float getY() {
		return y;
	}

	float getZ() {
		return z;
	}

	void setCombinedFlags(AbstractNode& a, AbstractNode& b) {
		nodeIdAndFlags = (a.nodeIdAndFlags | b.nodeIdAndFlags) & ML_COMBO_MASK;
		nodeIdAndFlags |= a.nodeIdAndFlags & b.nodeIdAndFlags & ML_CLAMP;
		//clamp only if both children use clamping
		if ((a.nodeIdAndFlags & ML_MATCH_MASK) != (b.nodeIdAndFlags
				& ML_MATCH_MASK)) {
			//throw new RuntimeException();
			std::cout << "Error in setCombinedFlags " << std::endl;
		}
	}

	virtual bool isLeaf()=0;

	virtual int size()=0;

};
int AbstractNode::globalNumReps = -1;
bool AbstractNode::globalMultitime = false;
std::vector<TimeVector> AbstractNode::repRandomNums;
float tempInit[] = { 1.0f };
const TimeVector AbstractNode::singleTimeVector(tempInit, tempInit
		+ sizeof(tempInit) / sizeof(float));
const int AbstractNode::ML_CLAMP = 0x02; //subject to clamping?
const int AbstractNode::ML_COMBO_MASK = 0xFC; //flags that should be shared with a parent
const int AbstractNode::ML_MATCH_MASK = 0xFC;
const int AbstractNode::ML_ID_SHIFT = 8;
#endif /* ABSTRACTNODE_H_ */
