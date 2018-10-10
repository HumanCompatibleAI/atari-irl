for dir in $(ls $1); do echo 'dir: ' $dir; for file in $(ls $1/$dir); do echo 'file: ' $file; grep -B 1 Return $1/$dir/$file | grep \|; done done
