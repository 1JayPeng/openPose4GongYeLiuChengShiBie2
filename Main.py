import sys

from PyQt5 import QtWidgets

from qt5Window import Ui_MainWindow

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

ui = Ui_MainWindow()
ui.setupUi(MainWindow)

MainWindow.show()
sys.exit(app.exec_())
