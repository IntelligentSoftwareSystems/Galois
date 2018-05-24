#ifndef VIEWER_H
#define VIEWER_H

#include <QGLViewer/qglviewer.h>
#include "Scene.h"

class Viewer: public QGLViewer {
  Scene& scene;
  unsigned refdelay;

  void initLights();

  void drawTime();

protected:
  virtual void draw() override;
  virtual void init() override;
  virtual void animate() override;
  virtual QString helpString() const override;

public:
  Viewer(Scene& s, unsigned refdelay): scene(s), refdelay (refdelay) { }
};

#endif
