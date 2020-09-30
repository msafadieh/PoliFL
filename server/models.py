from sqlalchemy import Column, ForeignKey, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Node(Base):
    __tablename__ = 'nodes'

    id = Column(Integer, primary_key=True)
    label = Column(String, unique=True)
    api_key = Column(String)
    public_key = Column(String)
    allowed_ips = Column(String)
    applications = relationship("Policy", back_populates="node", cascade="delete, delete-orphan")

    def __repr__(self):
        return f"<Node(label='{self.label}')>"


class Application(Base):
    __tablename__ = 'applications'

    id = Column(Integer, primary_key=True)
    api_key = Column(String)
    label = Column(String)
    nodes = relationship("Policy", back_populates="application", cascade="delete, delete-orphan")

class Policy(Base):
    __tablename__ = 'policies'

    value = Column(String)
    
    application_id = Column(ForeignKey('applications.id'), primary_key=True)
    node_id = Column(ForeignKey('nodes.id'), primary_key=True)
    
    application = relationship("Application", back_populates="nodes")
    node = relationship("Node", back_populates="applications")
