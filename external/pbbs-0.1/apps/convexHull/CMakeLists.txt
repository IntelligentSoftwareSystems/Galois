set(prefix convexHull)

inSphere(2DinSphere_10M 2 10000000)
onSphere(2DonSphere_10M 2 10000000)
kuzmin(2Dkuzmin_10M 2 10000000)

## 

include_directories(common)

add_subdirectory(quickHull)
add_subdirectory(serialHull)
