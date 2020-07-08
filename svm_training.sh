conda activate sonar
NOVELTIES=("A", "B", "C", "D")
for pca_comp in {1..100..3}; do
    for gamma in 0.01 0.1 1 10 100; do
        for C_param in 0.01 0.1 1 10 100; do
            for nov in "A" "B" "C" "D"; do
                python ~/Workspace/sonar/svm_kfold.py 1024 0 3 $nov 5 $C_param $gamma $pca_comp
            done
        done
    done
done