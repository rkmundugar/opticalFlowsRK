#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <future>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
   void frameSplitComp();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

private:
    void splitFrames(QString,QString);
    int lucas_kanade(QString, QString);
    template < typename Method, typename... Args>
    void dense_optical_flow(int,QString,QString,QString,Method,bool,Args&&...);

private:
    Ui::MainWindow *ui;
    std::future<void> future;
};
#endif // MAINWINDOW_H
