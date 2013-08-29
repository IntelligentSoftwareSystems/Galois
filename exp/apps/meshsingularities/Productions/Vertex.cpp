#include "Vertex.h"

Vertex::Vertex(Vertex *Left, Vertex *Right, Vertex *Parent, VertexType type, int systemSize)
{
    this->left = Left;
    this->right = Right;
    this->parent = Parent;
    this->type = type;
    this->system = new EquationSystem(systemSize);
}

Vertex::~Vertex()
{
    /*if (this->system != NULL) {
        delete this->system;
    } */

    if (this->left != NULL) {
        delete this->left;
    }
    if (this->right != NULL) {
        delete this->right;
    }

}
	
void Vertex::setLeft(Vertex *v) 
{
    this->left = v;
}
	
void Vertex::setRight(Vertex *v) 
{
    this->right = v;
}
	
void Vertex::setType(VertexType t)
{
    this->type = t;
}
