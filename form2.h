#ifndef FORM2_H
#define FORM2_H

#include <QMainWindow>
#include <future>
#include <opencv2/core/core.hpp>

namespace Ui {
class form2;
}

class form2 : public QMainWindow
{
    Q_OBJECT

public:
    explicit form2(QWidget *parent = nullptr);
    ~form2();

signals:
   void frameSplitComp();

private slots:
    //template <typename Method>
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

private:
    void splitFrames(QString,QString);
    template < typename Method, typename... Args>
    cv::Mat dense_optical_flow(QString,QString,QString,QString,Method,bool,Args&&...);

private:
    Ui::form2 *ui;
    std::future<void> future;
};

#endif // FORM2_H
