import tensorflow as tf
from matplotlib import pyplot as plt

testb = tf.data.Dataset.load("tf_saved/grub_test")
teste = tf.data.Dataset.load("tf_saved/grue_test")

grub = tf.keras.models.load_model("tf_saved/grub")
grue = tf.keras.models.load_model("tf_saved/grue")

print(len(testb))
test_results = []
for i, o in testb:
    pred = grub(i)
    test_results.append(tf.math.reduce_mean(tf.math.abs(pred - o)).numpy())

test_results = list(enumerate(test_results))
test_results.sort(key=lambda x: x[1])
print(test_results)

print(grub.evaluate(testb))
print(grue.evaluate(teste))

plt.subplots_adjust(hspace=0.5)
med = (len(testb) - 1) // 2
i, o = testb.take(1).get_single_element()
pred = grub(i)
axes = plt.subplot(3,1,1)
axes.scatter(range(1, 8), o[0, :, 0], label="Actual")
axes.scatter(range(1, 8), pred[0, :, 0], label="Predicted")
axes.legend()
axes.set_title("Best")
i, o = testb.skip(med).take(1).get_single_element()
pred = grub(i)
axes = plt.subplot(3,1,2)
axes.scatter(range(1, 8), o[0, :, 0], label="Actual")
axes.scatter(range(1, 8), pred[0, :, 0], label="Predicted")
axes.legend()
axes.set_title("Median")
i, o = testb.skip(len(testb) - 1).take(1).get_single_element()
pred = grub(i)
axes = plt.subplot(3,1,3)
axes.scatter(range(1, 8), o[0, :, 0], label="Actual")
axes.scatter(range(1, 8), pred[0, :, 0], label="Predicted")
axes.legend()
axes.set_title("Worst")
plt.show()
