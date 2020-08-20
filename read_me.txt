
#2020/08/11

代码目录："/home/songzhuoran/super_resolution/sr_brame_gen.py"

运行前指定classname变量，默认为'calendar'

运行所需文件：(当前目录："/home/songzhuoran/super_resolution/")
	idx文件："Info_BIx4/idx/"	  GetMotionVector.sh  
	mv文件："Info_BIx4/mvs/"          GetMotionVector.sh
	残差文件："Info_BIx4/Residuals/"  生成程序："GetResidual_csv.py"
	BIX4图片文件："Vid4/BIx4/"
	SR后图片文件："EDVR/results/Vid4/"

输出目录："bframe_sr_test/"(为保证正常输出，需在输出目录中预先创建calendar文件夹)

代码中有多个bframe_gen_kernel，使用不同的reonstruction方法，在DFS函数中可改变调用的kernel

推荐调用bframe_gen_kernel5
	342行cv2.resize函数中可改变对残差的插值方法。
	336行提供两种ref块来源(直接读SR文件/考虑误差传递)

如不添加残差，只使用mv，可调用bframe_gen_kernel4


