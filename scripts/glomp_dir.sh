for i in $(ls $1); do echo $i; grep -B 1 Return $1/$i | grep \|; done 
