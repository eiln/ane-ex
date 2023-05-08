
add:
	anecc hwx/add.hwx -c
	g++ -I/usr/include/libane -I/usr/include/eigen3 \
		add.anec.o add.cpp -o add.out /usr/lib/libane.a
srgan:
	anecc hwx/srgan.hwx -c
	g++ -I/usr/include/libane -I/home/eileen/d4 \
		srgan.anec.o srgan.cpp -o srgan.out /usr/lib/libane.a

clean:
	rm -f *.o *.so anec_*.h *.out
