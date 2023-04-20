{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype  == "class" %}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {% if methods %}
   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endif %}

   {% if attributes %}
   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}
   {% endif %}

   {% endblock %}

{% else %}

.. auto{{ objtype }}:: {{ objname }}

{% endif %}
