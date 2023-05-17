import os
import shutil

# needs to be in the same directory with manage.py
dir_path = os.path.dirname(os.path.abspath(__file__))
# django_project_name = os.path.basename(dir_path)
django_project_name = "sd_multi"

main_path = os.path.join(dir_path, django_project_name)
wsgi_path = os.path.join(main_path, "wsgi.py")
static_path = os.path.join(dir_path, "static")
venv_path = os.path.join(dir_path, "venv")

# note: AudoDL requires port 6006；need to change '/etc/apache2/ports.conf' as well
conf_raw = '''
<VirtualHost *:80>
    WSGIApplicationGroup %{GLOBAL}
    
    WSGIScriptAlias / {wsgi}
    
    Alias /static/ {static}/
    
    <Directory {static}>
        Require all granted
    </Directory>

    <Directory {main}>
    <Files wsgi.py>
        Require all granted
    </Files>
    </Directory>
</VirtualHost>

WSGIPythonHome {venv}
'''.format(GLOBAL = "{GLOBAL}", wsgi=wsgi_path, static=static_path, main=main_path, venv=venv_path)

# print(conf_raw)
conf_file = '{}.conf'.format(django_project_name)
with open(conf_file, 'w') as f:
    f.write(conf_raw)

try:
    os.remove("/etc/apache2/sites-available/"+conf_file)
except:
    pass
shutil.move(conf_file, "/etc/apache2/sites-available")

# authorize
# os.system("chmod -R 644 {}".format(main_path))
# os.system("find {} -type d | xargs chmod 755".format(main_path))

os.system("service apache2 reload")
os.system("a2dissite 000-default && a2ensite {}".format(django_project_name))
