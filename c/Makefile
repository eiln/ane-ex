
matmul:
	anecc hwx/matmul.hwx -c
	gcc -I. -I/usr/include/libane matmul.anec.o matmul.c -o matmul.out /usr/lib/libane.a

rsqrt:
	anecc hwx/rsqrt.hwx -c
	gcc -I. -I/usr/include/libane rsqrt.anec.o rsqrt.c -o rsqrt.out -lane

clean:
	rm -f *.o *.so anec_*.h *.out
