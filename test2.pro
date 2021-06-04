QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    form2.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    form2.h \
    mainwindow.h

FORMS += \
    form2.ui \
    mainwindow.ui

#LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_core451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_imgproc451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_highgui451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_imgcodecs451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_videoio451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_video451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_calib3d451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_photo451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_features2d451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_flann451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_objdetect451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_ml451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_tracking451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_videostab451.lib"
LIBS += "D:\\Software installations\\opencv\\opencv-build\\install\\x64\\vc15\\lib\\opencv_optflow451.lib"


INCLUDEPATH += "D:\\Software installations\\opencv\\opencv-build\\install\\include"

DEPENDPATH += "D:\\Software installations\\opencv\\opencv-build\\install\\include"


INCLUDEPATH += "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.17763.0\\ucrt"
LIBS += -L"C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.17763.0\\ucrt\\x64"
#LIBS += "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10586.0\um\x64\shell32.lib"

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.17763.0/um/x64/' -lshell32
else:win32:CONFIG(debug, debug|release): LIBS += -L'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.17763.0/um/x64/' -lshell32
else:unix: LIBS += -L'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.17763.0/um/x64/' -lshell32

INCLUDEPATH += 'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.17763.0/um/x64'
DEPENDPATH += 'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.17763.0/um/x64'
