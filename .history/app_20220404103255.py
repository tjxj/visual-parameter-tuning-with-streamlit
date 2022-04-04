
# print(y_pro_test)

# y_pro_test

fpr, tpr, thresholds = roc_curve(y_test, y_pro_test[:, -1])

print('auc: ', (roc_auc_score(y_test, y_pro_test[:, -1])))

# 绘制auc曲线

# 绘制面积

plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')

# 绘制曲线

plt.plot(fpr, tpr, color='black', lw=1)

# 添加对角线

plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# 添加文本信息

plt.text(0.5, 0.3, 'ROC curve (area=%0.2f)' % roc_auc_score(y_test, y_pro_test[:, -1]))

# 添加x轴和y轴

plt.xlabel('1-Specificity')

plt.ylabel('Sensitivity')

# 显示图形

plt.show()

# 计算ks，至少要大于0.2

print("ks: ", max(tpr - fpr))

print('------------------------------------')

print(classification_report(y_test, np.where(y_pro_test[:, -1] > 0.5, 1, 0)))

print('-----------------混淆矩阵-----------------------')

print(confusion_matrix(y_test, np.where(y_pro_test[:, -1] > 0.5, 1, 0)))

# 查看pr曲线

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pro_test[:, -1])

plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

plt.xlabel("Threshold")

plt.legend(loc="upper left")

plt.ylim([0, 1])

plt.grid(True, linestyle="--", alpha=0.5)

fpr, tpr, thresholds = roc_curve(y_test, y_pro_test[:, -1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.axis([0, 1, 0, 1])
    
    plt.xlabel('False Positive Rate')
    
    plt.ylabel('True Positive Rate')
    
    # plot_roc_curve(fpr, tpr)
    
    plt.show()