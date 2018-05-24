#include "Viewer.h"

void Viewer::drawTime() {
  char buf[1024];
  snprintf(buf, 1024, "SimTime: %.2f", scene.time());

  QFont font;

  font.setPointSize(25);
  glDisable(GL_LIGHTING);
  glColor3f(0.0, 1.0, 1.0);
  // drawText(width() - 300, height() - 20, buf, font);
  drawText (100, 25, buf, font);
  glEnable(GL_LIGHTING);
}

void Viewer::draw() {
  drawTime();
  scene.draw(this);
}

void Viewer::initLights() {
  // Ambient light
  glEnable(GL_LIGHT0);
  if (false) {
    // Spot light
    glEnable(GL_LIGHT1);

    GLfloat ambient[4] = {0.8f, 0.2f, 0.2f, 1.0f};
    GLfloat diffuse[4] = {1.0f, 0.4f, 0.4f, 1.0f};
    GLfloat specular[4] = {1.0f, 0.0f, 0.0f, 1.0f};

    glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 3.0);
    glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 20.0);
    glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 0.5f);
    glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 1.0f);
    glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 1.5f);
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
  }
  if (true) {
    // Directional light
    glEnable(GL_LIGHT2);

    GLfloat ambient[4] = {0.2f, 0.2f, 2.0f, 1.0f};
    GLfloat diffuse[4] = {0.8f, 0.8f, 1.0f, 1.0f};
    GLfloat specular[4] = {0.0f, 0.0f, 1.0f, 1.0f};

    glLightfv(GL_LIGHT2, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT2, GL_SPECULAR, specular);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse);

    GLfloat pos[4] = {0.5f, 0.5f, 0.0f, 0.0f};
    glLightfv(GL_LIGHT2, GL_POSITION, pos);
  }
}

void Viewer::animate() {
  scene.animate();
}

void Viewer::init() {
  initLights();

  qglviewer::Vec sceneDim = scene.init();

  restoreStateFromFile();

  float largest = std::max(std::max(sceneDim[0], sceneDim[1]), sceneDim[2]); 
  setSceneRadius(largest / 2);
  setSceneCenter(sceneDim / 2);
  camera()->showEntireScene();
  
  setAnimationPeriod(refdelay);
  //help();
  //startAnimation();
}

QString Viewer::helpString() const {
  QString text("<h2>S i m p l e V i e w e r</h2>");
  text += "Use the mouse to move the camera around the object. ";
  text += "You can respectively revolve around, zoom and translate with the three mouse buttons. ";
  text += "Left and middle buttons pressed together rotate around the camera view direction axis<br><br>";
  text += "Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
  text += "Simply press the function key again to restore it. Several keyFrames define a ";
  text += "camera path. Paths are saved when you quit the application and restored at next start.<br><br>";
  text += "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
  text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. ";
  text += "See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>";
  text += "Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). ";
  text += "A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>";
  text += "A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. ";
  text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";
  text += "Press <b>Escape</b> to exit the viewer.";
  return text;
}
