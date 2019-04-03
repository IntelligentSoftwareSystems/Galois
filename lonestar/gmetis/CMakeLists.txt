app(gmetis)

add_test_scale(small1 gmetis "${BASEINPUT}/reference/structured/rome99.gr" 4)
add_test_scale(small2 gmetis "${BASEINPUT}/scalefree/rmat16-2e10-a=0.57-b=0.19-c=0.19-d=.05.gr" 256)
#add_test_scale(web gmetis "${BASEINPUT}/road/USA-road-d.USA.gr" 256)
