class Dataset_manager:
    def __init__(self, pref, pref2, 
             tra_data_path, 
             con_data_path, 
             bg_val_data_path, 
             sg_val_data_path, 
             REVERSE=False,
             tra_data_field=None,
             con_data_field=None,
             bg_val_data_field=None,
             sg_val_data_field=None):
        if REVERSE:
            self.pref=pref2 
            self.perf2=pref
            self.tra_data_path=con_data_path
            self.con_data_path=tra_data_path
            self.bg_val_data_path=sg_val_data_path
            self.sg_val_data_path=bg_val_data_path
            self.REVERSE=REVERSE
            self.tra_data_field=con_data_field
            self.con_data_field=tra_data_field
            self.bg_val_data_field=sg_val_data_field
            self.sg_val_data_field=bg_val_data_field
        else:
            self.pref=pref 
            self.pref2=pref2
            self.tra_data_path=tra_data_path
            self.con_data_path=con_data_path
            self.bg_val_data_path=bg_val_data_path
            self.sg_val_data_path=sg_val_data_path
            self.REVERSE=REVERSE
            self.tra_data_field=tra_data_field
            self.con_data_field=con_data_field
            self.bg_val_data_field=bg_val_data_field
            self.sg_val_data_field=sg_val_data_field
            
    def loadIN_X_tra:
    def loadIN_X_
        