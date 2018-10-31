cd tensorflow-for-poets-2

python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb \
    --image=tf_files/boba.jpg
