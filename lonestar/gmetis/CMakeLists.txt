app(gmetis)

add_test_scale(small gmetis "${BASEINPUT}/structured/rome99.gr" 4)
add_test_scale(web gmetis "${BASEINPUT}/road/USA-road-d.USA.gr" 256)
