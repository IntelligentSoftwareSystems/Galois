app_analy_gpu(pta pointstoanalysis)
add_test_gpu(pointstoanalysis tshark tshark.out pta ${BASEINPUT}/java/pta/tshark_nodes.txt ${BASEINPUT}/java/pta/tshark_constraints_after_hcd.txt ${BASEINPUT}/java/pta/tshark_hcd.txt ${BASEINPUT}/java/pta/tshark_correct_soln_001.txt)
